"use client";

import { useEffect, useState } from "react";
import type { ConfigFieldSchema } from "@/lib/api";

interface ConfigFieldRowProps {
  field: ConfigFieldSchema;
  onOverride: (path: string, value: unknown) => void;
  onRevert: (path: string) => void;
  disabled?: boolean;
}

function formatValue(val: unknown): string {
  if (val === undefined || val === null) return "—";
  if (typeof val === "boolean") return val ? "true" : "false";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}

export function ConfigFieldRow({
  field,
  onOverride,
  onRevert,
  disabled = false,
}: ConfigFieldRowProps) {
  const [localValue, setLocalValue] = useState<string>(() => formatValue(field.current_value));
  const [pending, setPending] = useState(false);
  const overridden = field.overridden ?? false;

  useEffect(() => {
    setLocalValue(formatValue(field.current_value));
  }, [field.current_value]);

  const handleApply = () => {
    if (field.remote_excluded) return;
    let value: unknown;
    if (field.type === "bool") {
      value = localValue.toLowerCase() === "true" || localValue === "1";
    } else if (field.type === "int") {
      value = parseInt(localValue, 10);
      if (Number.isNaN(value)) return;
    } else if (field.type === "float") {
      value = parseFloat(localValue);
      if (Number.isNaN(value)) return;
    } else {
      value = localValue;
    }
    setPending(true);
    onOverride(field.path, value);
    setPending(false);
  };

  const handleRevert = () => {
    setPending(true);
    onRevert(field.path);
    setPending(false);
  };

  if (field.remote_excluded) return null;

  const isBool = field.type === "bool";
  const isNum = field.type === "int" || field.type === "float";
  const hasEnum = field.enum_values && field.enum_values.length > 0;
  // Ensure select value is one of the options so we never send invalid_enum_value
  const enumValue =
    hasEnum && field.enum_values!.includes(localValue)
      ? localValue
      : hasEnum
        ? field.enum_values![0]
        : "";

  return (
    <div className="config-field-row">
      <div className="config-field-meta">
        <code className="config-field-path">{field.path}</code>
        <span className="config-field-type">{field.type}</span>
        {overridden && (
          <span className="config-field-badge" title="Overridden (differs from baseline)">
            overridden
          </span>
        )}
      </div>
      <div className="config-field-values">
        <div className="config-field-baseline" title="Value from code">
          <span className="config-field-label">Baseline</span>
          <span className="config-field-value">{formatValue(field.baseline_value)}</span>
        </div>
        <div className="config-field-current" title="Effective value">
          <span className="config-field-label">Current</span>
          {isBool ? (
            <label className="config-field-toggle">
              <input
                type="checkbox"
                checked={field.current_value === true}
                onChange={(e) => {
                  setLocalValue(e.target.checked ? "true" : "false");
                  onOverride(field.path, e.target.checked);
                }}
                disabled={disabled}
              />
              <span>{formatValue(field.current_value)}</span>
            </label>
          ) : hasEnum ? (
            <select
              className="config-field-select"
              value={enumValue}
              onChange={(e) => {
                const v = e.target.value;
                setLocalValue(v);
                onOverride(field.path, v);
              }}
              disabled={disabled}
              aria-label={`Select ${field.name}`}
            >
              {field.enum_values!.map((v) => (
                <option key={v} value={v}>
                  {v.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          ) : (
            <input
              type={isNum ? "number" : "text"}
              className="config-field-input"
              value={localValue}
              onChange={(e) => setLocalValue(e.target.value)}
              onBlur={handleApply}
              onKeyDown={(e) => e.key === "Enter" && handleApply()}
              disabled={disabled}
            />
          )}
        </div>
      </div>
      <div className="config-field-actions">
        {!isBool && !hasEnum && (
          <button
            type="button"
            className="config-field-btn config-field-btn-apply"
            onClick={handleApply}
            disabled={disabled || pending}
          >
            Apply
          </button>
        )}
        {overridden && (
          <button
            type="button"
            className="config-field-btn config-field-btn-revert"
            onClick={handleRevert}
            disabled={disabled || pending}
            title="Revert to baseline"
          >
            Revert
          </button>
        )}
      </div>
    </div>
  );
}
