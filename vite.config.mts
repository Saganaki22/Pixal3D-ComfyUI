import { defineConfig } from "vite";

export default defineConfig({
  build: {
    emptyOutDir: false,
    outDir: "web",
    lib: {
      entry: "src/pixal3d_camera_control.ts",
      formats: ["es"],
      fileName: () => "pixal3d_camera_control.js"
    },
    rollupOptions: {
      external: [/^\/scripts\//]
    }
  }
});
