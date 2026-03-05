"use client";

import { useCallback, useEffect, useState } from "react";
import { AgentIcon } from "./AgentIcon";
import { ConfigSectionBlock } from "./ConfigSectionBlock";
import { fetchAgentConfig, patchAgentConfig, type AgentConfigResponse } from "@/lib/api";

interface AgentConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiBase: string;
  agentName?: string;
  agentDisplayName?: string;
}

export function AgentConfigModal({
  isOpen,
  onClose,
  apiBase,
  agentName,
  agentDisplayName,
}: AgentConfigModalProps) {
  const [config, setConfig] = useState<AgentConfigResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patchStatus, setPatchStatus] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const load = useCallback(async () => {
    if (!apiBase) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAgentConfig(apiBase, agentName);
      if (data) {
        setConfig(data);
        setError(null);
      } else {
        setError("Remote config not available (GET /config failed or not supported)");
        setConfig(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load config");
      setConfig(null);
    } finally {
      setLoading(false);
    }
  }, [apiBase, agentName]);

  useEffect(() => {
    if (isOpen) {
      load();
      setPatchStatus(null);
    }
  }, [isOpen, load]);

  const handleOverride = useCallback(
    async (path: string, value: unknown) => {
      if (!config) return;
      setPatchStatus(null);
      const result = await patchAgentConfig(apiBase, config.agent_id, [{ path, value }], agentName);
      if (!result) {
        setPatchStatus({ type: "error", message: "PATCH request failed" });
        return;
      }
      const acceptedSet = new Set(result.accepted);
      const rejectedOnly = result.rejected.filter(([p]) => !acceptedSet.has(p));
      if (rejectedOnly.length > 0) {
        setPatchStatus({
          type: "error",
          message: rejectedOnly.map(([p, r]) => `${p}: ${r}`).join("; "),
        });
        return;
      }
      setPatchStatus({ type: "success", message: "Updated" });
      await load();
    },
    [apiBase, agentName, config, load]
  );

  const handleRevert = useCallback(
    async (path: string) => {
      if (!config) return;
      setPatchStatus(null);
      const result = await patchAgentConfig(
        apiBase,
        config.agent_id,
        [{ path, value: null }],
        agentName
      );
      if (!result) {
        setPatchStatus({ type: "error", message: "Revert request failed" });
        return;
      }
      const acceptedSet = new Set(result.accepted);
      const rejectedOnly = result.rejected.filter(([p]) => !acceptedSet.has(p));
      if (rejectedOnly.length > 0) {
        setPatchStatus({
          type: "error",
          message: rejectedOnly.map(([p, r]) => `${p}: ${r}`).join("; "),
        });
        return;
      }
      setPatchStatus({ type: "success", message: "Reverted to baseline" });
      await load();
    },
    [apiBase, agentName, config, load]
  );

  if (!isOpen) return null;

  const displayName = agentDisplayName ?? agentName ?? config?.agent_name ?? "Agent";

  return (
    <>
      <div className="modal-overlay" onClick={onClose} aria-hidden="true" />
      <aside
        className="agent-details-modal agent-config-modal"
        role="dialog"
        aria-label="Agent config"
      >
        <div className="modal-header">
          <div className="modal-title-row">
            <AgentIcon size={22} />
            <div className="modal-title-wrap">
              <h2 className="modal-agent-name">{displayName}</h2>
              <span className="modal-subtitle">Remote config</span>
            </div>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="modal-body config-modal-body">
          {loading && !config && <p className="config-loading">Loading config…</p>}
          {error && (
            <p className="config-error">
              {error}
              <button type="button" className="config-retry" onClick={load}>
                Retry
              </button>
            </p>
          )}
          {patchStatus && (
            <div
              className={
                patchStatus.type === "success"
                  ? "config-status config-status-success"
                  : "config-status config-status-error"
              }
            >
              {patchStatus.message}
            </div>
          )}
          {config && !loading && (
            <>
              <p className="config-hint">
                Baseline = values from code. Override a field and click Apply, or Revert to restore
                baseline.
              </p>
              <div className="config-sections">
                {Object.entries(config.sections).map(([key, section]) => (
                  <ConfigSectionBlock
                    key={key}
                    sectionKey={key}
                    section={section}
                    onOverride={handleOverride}
                    onRevert={handleRevert}
                    disabled={loading}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  );
}
