import { ReactNode } from "react";
import { TabId } from "../lib/uiState";

type AppShellProps = {
  activeTab: TabId;
  adminEnabled: boolean;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
};

export function AppShell({ activeTab, adminEnabled, onTabChange, children }: AppShellProps) {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="brand-title">Doedutch</h1>
          <p className="brand-subtitle">Minimal UI (React + Vite)</p>
        </div>
      </header>

      <nav className="tabs" aria-label="Main tabs">
        <button
          className={`tab-btn ${activeTab === "generate" ? "active" : ""}`}
          onClick={() => onTabChange("generate")}
          type="button"
        >
          Generate
        </button>
        <button
          className={`tab-btn ${activeTab === "settings" ? "active" : ""}`}
          onClick={() => onTabChange("settings")}
          type="button"
        >
          Settings
        </button>
        {adminEnabled && (
          <button
            className={`tab-btn ${activeTab === "admin" ? "active" : ""}`}
            onClick={() => onTabChange("admin")}
            type="button"
          >
            Admin
          </button>
        )}
      </nav>

      {children}
    </div>
  );
}
