"use client";

import { useState } from "react";
import { ConfigFieldRow } from "./ConfigFieldRow";
import type { ConfigFieldSchema, ConfigSectionSchema } from "@/lib/api";

interface ConfigSectionBlockProps {
  sectionKey: string;
  section: ConfigSectionSchema;
  onOverride: (path: string, value: unknown) => void;
  onRevert: (path: string) => void;
  disabled?: boolean;
}

function collectFields(fields: ConfigFieldSchema[]): ConfigFieldSchema[] {
  const out: ConfigFieldSchema[] = [];
  for (const f of fields) {
    if (!f.remote_excluded) out.push(f);
    if (f.children?.length) out.push(...collectFields(f.children));
  }
  return out;
}

export function ConfigSectionBlock({
  sectionKey,
  section,
  onOverride,
  onRevert,
  disabled = false,
}: ConfigSectionBlockProps) {
  const [open, setOpen] = useState(true);
  const fields = collectFields(section.fields);
  if (fields.length === 0) return null;

  return (
    <div className="config-section-block">
      <button
        type="button"
        className="config-section-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="config-section-title">
          {sectionKey}
          <span className="config-section-class">{section.class_name}</span>
        </span>
        <span className="config-section-toggle">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="config-section-fields">
          {fields.map((field) => (
            <ConfigFieldRow
              key={field.path}
              field={field}
              onOverride={onOverride}
              onRevert={onRevert}
              disabled={disabled}
            />
          ))}
        </div>
      )}
    </div>
  );
}
