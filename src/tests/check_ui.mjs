// Fails if the inline UI script has a syntax error. Usage: node tests/check_ui.mjs
import { readFileSync } from "node:fs";
import { Script } from "node:vm";

const html = readFileSync(new URL("../app/static/index.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (!scripts.length) throw new Error("no inline <script> found in index.html");
scripts.forEach((code, i) => new Script(code, { filename: `index.html#script${i}` }));
console.log(`UI script OK (${scripts.length} block${scripts.length > 1 ? "s" : ""})`);
