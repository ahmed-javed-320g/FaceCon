# Building the real-genotype panel from 1000 Genomes

What this gives us: **real human genotypes** (real allele frequencies, real
linkage between neighbouring SNPs, real continuous ancestry) with **simulated
faces**. It is not real face data and no result from it is a statement about
real faces. It removes one specific criticism of the synthetic study — that
unlinked, uniform-frequency SNPs make the problem artificially easy — and
nothing else.

Source: 1000 Genomes 30x high-coverage NYGC callset on GRCh38, 3,202 samples,
dbSNP-156 rsIDs, as distributed on the PLINK 2 resources page
(Byrska-Bishop et al. 2022, *Cell* 185:3426).

---

## 0. Files you should already have in `D:\FaceCon\1000G`

| File | Note |
|---|---|
| `all_hg38.pgen` | decompressed, ~9.3 GB |
| `all_hg38.pvar.zst` | leave compressed (the `vzs` modifier reads it) |
| `all_hg38.psam` | sample table: sex + population + super-population |
| `deg2_hg38.king.cutoff.out.id` | 2nd-degree relatives to drop |
| `plink2.exe` | PLINK 2 Windows AVX2 |

Also copy `data_prep/named_snps.txt` (54 rsIDs) from the repo into this folder.

Disk needed for the outputs: about 1 GB. Nothing here overwrites the downloads.

---

## 1. Open PowerShell in that folder

```powershell
cd D:\FaceCon\1000G
```

Every command below is one line. `.\plink2.exe` — the `.\` is required in
PowerShell.

---

## 2. Pass 1 — list the SNPs that pass the filters

```powershell
.\plink2.exe --pfile all_hg38 vzs --remove deg2_hg38.king.cutoff.out.id --chr 1-22,X --snps-only just-acgt --max-alleles 2 --maf 0.01 --geno 0.02 --rm-dup exclude-all --exclude named_snps.txt --write-snplist --out kg_common
```

What each filter is for:

- `--remove deg2...` drops 1st- and 2nd-degree relatives, so the people are
  unrelated (related people inflate any apparent prediction accuracy). About
  **2,573 people** should remain.
- `--chr 1-22,X` autosomes plus X (the HIrisPlex-S panel has no X markers, but
  two Shaffer SNPs are near the X; keeping X also lets the sex check work).
- `--snps-only just-acgt --max-alleles 2` plain biallelic SNPs only, no indels,
  no multi-allelic sites — our genotype matrix is one dosage number per SNP.
- `--maf 0.01` drops SNPs too rare to learn anything from in 2,573 people.
- `--geno 0.02` drops SNPs missing in more than 2% of people.
- `--rm-dup exclude-all` drops any rsID that appears more than once, so SNP IDs
  are unique keys.
- `--exclude named_snps.txt` keeps the random panel disjoint from the 54 named
  pigmentation/face SNPs, so a named SNP can never also be drawn as a random
  "causal" SNP.

Expect roughly 6–8 million SNPs in `kg_common.snplist`. Report the exact number.

---

## 3. Choose 100,000 of them, reproducibly

Run this from the **repo** folder (the one with `data_prep\`), pointing at the
snplist:

```powershell
python data_prep\select_snps.py D:\FaceCon\1000G\kg_common.snplist D:\FaceCon\1000G\kg_selected.txt 100000
```

Fixed seed 20261016, so anyone re-running this gets the same 100,000 SNPs.
Two passes (list, then choose, then extract) rather than `--thin-count` so
there is no ambiguity about what PLINK filtered in which order.

---

## 4. Pass 2 — write the two genotype files

Genome-wide panel:

```powershell
cd D:\FaceCon\1000G
.\plink2.exe --pfile all_hg38 vzs --remove deg2_hg38.king.cutoff.out.id --extract kg_selected.txt --make-bed --out kg_main
```

The named SNPs — **no MAF or missingness filter here**, because the published
HIrisPlex-S model needs these exact loci whether they are common or not:

```powershell
.\plink2.exe --pfile all_hg38 vzs --remove deg2_hg38.king.cutoff.out.id --extract named_snps.txt --make-bed --out kg_named
```

Both use the same `--remove` and no other sample filter, so both files hold the
same people in the same order — `build_1kg.py` checks this and stops if not.

Expect `kg_main.bed` ≈ 64 MB (2,573 × 100,000 at 2 bits per genotype) and
`kg_named.bim` to have 40–54 of the 54 rsIDs. Some will be missing: the panel is
WGS, so most should be present, but any that are multi-allelic in 1000G or
absent get reported by name in step 5. That is expected and handled — a missing
marker is filled with the reference allele count and listed in the report.

---

## 5. Build the genotype archive

From the repo folder:

```powershell
python scripts\build_1kg.py --dir D:\FaceCon\1000G --out outputs\kg_geno
```

This writes `outputs\kg_geno\genotypes_1kg.npz` and
`outputs\kg_geno\build_report.json`, and prints three checks:

1. **Sample counts per super-population** (AFR, AMR, EAS, EUR, SAS) and sex
   counts. Sanity: ~660 AFR, ~340 AMR, ~500 EAS, ~500 EUR, ~480 SAS.
2. **Allele-frequency check.** For each of the 41 HIrisPlex-S markers it prints
   the published European minor-allele frequency next to the frequency of the
   same effect allele in the 1000G EUR samples. A line marked `<-- check` means
   they differ by more than 0.15, which usually means the allele got flipped
   somewhere. A handful of flags is normal (the published frequencies come from
   different European samples); many flags means something is wrong with the
   allele alignment and should be reported before going further.
3. **HIrisPlex-S predictions by super-population.** Blue eyes and light skin
   should come out far commoner in EUR than in AFR or EAS. This is a
   plausibility check on the wiring, **not** an accuracy measurement — 1000G has
   no recorded eye or hair colour to check against.

---

## 6. Generate the dataset and run the experiments

See the commands in the message that accompanies this file, or:

```powershell
python scripts\generate_matched.py config\kg.yaml
python -m gpu.controls --config config\kg.yaml --out outputs\controls_kg.json
python -m gpu.controls --config config\kg.yaml --hide-causal --out outputs\controls_kg_hidden.json
python -m gpu.run_all --data-dir outputs\kg
```

`generate_matched.py` takes the config as a **positional** argument, not
`--config`. `gpu.controls` builds its own data from the config, so it can run
before the dataset exists; `gpu.run_all` reads `outputs\kg\dataset.npz` and
needs `generate_matched.py` to have run first.

### Two things the first real run turned up

- **Variant count comes out slightly over 100,000** (100,009 on the first run).
  Pass 2 has no `--snps-only`/`--max-alleles`, so a handful of rsIDs whose
  duplicate partners were filtered out in pass 1 bring those partners back.
  Harmless — the panel is indexed by column, not by rsID. For an exactly round
  number, add `--snps-only just-acgt --max-alleles 2 --rm-dup exclude-all` to the
  `kg_main` command.
- **Samples with unrecorded sex** are assigned female. The 1000G psam has one
  such sample in 3,202. Recorded in `build_report.json`.

RAM: the controls ladder holds the residualised genotype matrix in double
precision, about 2 GB at 2,573 × 100,000, and peaks around 6–8 GB overall. 16 GB
is comfortable. If it runs out, lower `n_snps` in the selection step to 50,000.

---

## Honesty note for the report

The manifest of the generated dataset records
`mode: real_genotypes_1000G_simulated_faces`. Every number produced from it
must be labelled that way. The faces, ages, heights and weights are simulated;
the genotype-to-face mapping is one we imposed. What is real is the genotype
matrix, the ancestry structure and the pigmentation marker genotypes.
