# FaceCon — deployable web bundle

A self-contained static site. **No build step, no server-side code, no API keys.**
Any static host serves it as-is.

```
index.html        landing page (links the two apps)
prototype.html    interactive prototype (pick subject → reveal → metrics)
holo.html         hologram reveal view
three.min.js      3D library (vendored locally — no CDN/network needed)
data/web_data.json   the model + demo data the apps read (~3.9 MB)
```

The apps `fetch()` `data/web_data.json`, so they must be opened over **http(s)**,
not `file://`. Any host below handles that automatically.

## Deploy (pick one — all free)

**Netlify Drop (easiest, no account, ~30 s):** go to https://app.netlify.com/drop
and drag this whole folder onto the page. You get a live URL instantly.

**Vercel (named in the brief):**
```bash
npm i -g vercel
cd facecon-deploy      # this folder
vercel                 # log in, accept defaults → preview URL
vercel --prod          # production URL to use in the demo
```
No `vercel.json` is needed — Vercel serves a static folder with zero config.

**GitHub Pages / Cloudflare Pages** also work — push the folder to a repo and point
the host at the root.

See `FaceCon_Hosting_Guide.md` for full step-by-step instructions and how to put the
link into the presentation.
