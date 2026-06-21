// The five relationship classes — a SEMANTIC legend, kept separate from the purple chrome theme.
// Order MUST mirror internal/label.Classes / ml.config.CLASS_NAMES (plans 03/04); never reorder.

export const CLASS_NAMES = [
  "MATERIAL_CONFLICT",
  "VERBAL_CONFLICT",
  "MATERIAL_COOPERATION",
  "VERBAL_COOPERATION",
  "STATUS_QUO",
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

export const CLASS_COLORS: Record<ClassName, string> = {
  MATERIAL_CONFLICT: "#ef4444", // red
  VERBAL_CONFLICT: "#f59e0b", // amber
  MATERIAL_COOPERATION: "#22c55e", // green
  VERBAL_COOPERATION: "#38bdf8", // sky
  STATUS_QUO: "#6b7280", // gray
};

export function classColor(name?: string | null): string {
  if (name && name in CLASS_COLORS) return CLASS_COLORS[name as ClassName];
  return CLASS_COLORS.STATUS_QUO;
}

// "MATERIAL_COOPERATION" -> "material cooperation"
export function classLabel(name: string): string {
  return name.toLowerCase().replace(/_/g, " ");
}
