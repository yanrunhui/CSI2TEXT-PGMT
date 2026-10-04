import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const {
  SKILL_DIR,
  TMP_DIR,
  RUNTIME_PYTHON,
  WORKSPACE_DIR,
  FINAL_PPTX,
  SOURCE_SVG,
} = process.env;

for (const [name, value] of Object.entries({
  SKILL_DIR,
  TMP_DIR,
  RUNTIME_PYTHON,
  WORKSPACE_DIR,
  FINAL_PPTX,
  SOURCE_SVG,
})) {
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must be an absolute path`);
  }
}

const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const svgBytes = await fs.readFile(SOURCE_SVG);
const presentation = Presentation.create({
  slideSize: { width: 1280, height: 800 },
});
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

slide.images.add({
  blob: svgBytes,
  contentType: "image/svg+xml",
  alt: "CSI–Text–Prototype Semantic Alignment with orange semantic-alignment arrows",
  fit: "contain",
  position: { left: 0, top: 0, width: 1280, height: 800 },
});

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(
  path.join(TMP_DIR, "slide-1.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const result = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: WORKSPACE_DIR,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: [
    "--expected-slide-size-emu",
    "12192000,7620000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, result }, null, 2));
