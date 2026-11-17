# FaceCon: deployable web bundle

An identical copy of `app/`, for static hosting. No build step, no server code, no
API keys. The only third-party code is three.js r147 (MIT) in `vendor/`.

```
index.html               demonstrator: Explore, Results, Pipeline
holo.html                hologram reveal
data/demo.js             mesh, 36 held-out people, recorded results (2.8 MB)
instrument-sans.woff2    typeface (SIL Open Font License, licence beside it)
vendor/                  three.js r147, OrbitControls, RoomEnvironment (MIT, licence beside them)
```

The data is a script rather than JSON, so the pages also work opened straight from
disk (`file://`). No local server is needed.

Refresh this folder after `python scripts/export_demo.py`:

```bash
cp app/index.html app/holo.html deploy/ && cp app/data/demo.js deploy/data/ && cp -r app/vendor deploy/
```

## Hosting (all free, all serve a static folder as-is)

- **Netlify Drop:** drag this folder onto https://app.netlify.com/drop.
- **Vercel:** `npx vercel` from this folder, then `npx vercel --prod`.
- **GitHub Pages or Cloudflare Pages:** point the host at this folder.

Every face shown is simulated, and the page says so. Do not host it anywhere that
would present it as a prediction about a real person.
