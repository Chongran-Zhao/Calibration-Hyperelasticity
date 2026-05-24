import React, { useEffect, useMemo, useState } from "react"
import { createRoot } from "react-dom/client"
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  Info,
  Layers,
  LineChart,
  RotateCcw,
  Save,
  Search,
  Settings,
  SlidersHorizontal,
  ZoomIn,
} from "lucide-react"
import "./styles.css"

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"

const samplePoints = [
  { x: 1.0, y: 0.0 },
  { x: 1.1, y: 0.18 },
  { x: 1.22, y: 0.48 },
  { x: 1.36, y: 0.92 },
  { x: 1.5, y: 1.55 },
  { x: 1.68, y: 2.48 },
  { x: 1.86, y: 3.75 },
]

const modeOptions = [
  {
    family: "CSS",
    label: "Compound Simple Shear",
    dot: "bg-cyan-600",
  },
  {
    family: "UT",
    label: "Uniaxial Tension",
    dot: "bg-blue-600",
  },
  {
    family: "PS",
    label: "Pure Shear",
    dot: "bg-emerald-600",
  },
  {
    family: "ET",
    label: "Equibiaxial Tension",
    dot: "bg-red-500",
  },
  {
    family: "SS",
    label: "Simple Shear",
    dot: "bg-teal-600",
  },
  {
    family: "BT",
    label: "Biaxial Tension",
    dot: "bg-violet-600",
  },
]

const iconMap = {
  analytics: LineChart,
  arrow_back: ArrowLeft,
  arrow_forward: ArrowRight,
  check_circle: CheckCircle2,
  chevron_right: ChevronRight,
  dns: Database,
  help: CircleHelp,
  info: Info,
  menu_book: BookOpen,
  monitoring: Activity,
  restart_alt: RotateCcw,
  save: Save,
  schema: Layers,
  science: FlaskConical,
  search: Search,
  settings: Settings,
  tune: SlidersHorizontal,
  zoom_in: ZoomIn,
}

function Icon({ children, className = "" }) {
  const Component = iconMap[children] ?? Info
  return <Component aria-hidden="true" className={`inline-block h-[1em] w-[1em] shrink-0 ${className}`} />
}

function App() {
  const [datasets, setDatasets] = useState([])
  const [author, setAuthor] = useState("")
  const [modes, setModes] = useState([])
  const [preview, setPreview] = useState({
    points: samplePoints,
    axes: { x: "Stretch lambda (-)", y: "Nominal stress P11 (MPa)" },
    metadata: {
      rows: samplePoints.length,
      source: "Waiting for database",
      selectedMode: "Uniaxial Tension",
      stressType: "First PK",
      setCount: 1,
    },
  })
  const [apiState, setApiState] = useState("Connecting")

  useEffect(() => {
    fetch(`${API_BASE}/api/datasets`)
      .then((res) => res.json())
      .then((data) => {
        const authors = data.authors ?? []
        setDatasets(authors)
        const preferred = authors.find((item) => item.author === "James_1975") ?? authors[0]
        if (preferred) {
          setAuthor(preferred.author)
          const fallback = preferred.modes?.find((item) => item.family === "UT") ?? preferred.modes?.[0]
          setModes(fallback ? [fallback.key] : [])
        }
        setApiState("Ready")
      })
      .catch(() => setApiState("Offline"))
  }, [])

  useEffect(() => {
    if (!author || !modes.length) return
    const params = new URLSearchParams({ author })
    modes.forEach((item) => params.append("mode", item))
    fetch(`${API_BASE}/api/preview?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        setPreview(data)
      })
      .catch(() => setApiState("Preview error"))
  }, [author, modes])

  const authorRecord = useMemo(() => datasets.find((item) => item.author === author), [author, datasets])
  const authorModes = authorRecord?.modes ?? []
  const selectedModeRecords = useMemo(
    () => modes.map((key) => authorModes.find((item) => item.key === key)).filter(Boolean),
    [authorModes, modes],
  )
  const primaryModeMeta = useMemo(() => {
    const family = selectedModeRecords[0]?.family ?? "UT"
    return modeOptions.find((item) => item.family === family) ?? modeOptions[0]
  }, [selectedModeRecords])

  useEffect(() => {
    if (!authorModes.length) return
    const validKeys = new Set(authorModes.map((item) => item.key))
    const validSelection = modes.filter((key) => validKeys.has(key))
    if (!validSelection.length) {
      const fallback = authorModes.find((item) => item.family === "UT") ?? authorModes[0]
      setModes([fallback.key])
      return
    }
    if (validSelection.join("|") !== modes.join("|")) {
      setModes(validSelection)
    }
  }, [author, authorModes, modes])

  function handleAuthorChange(value) {
    setAuthor(value)
    const record = datasets.find((item) => item.author === value)
    const fallback = record?.modes?.find((item) => item.family === "UT") ?? record?.modes?.[0]
    setModes(fallback ? [fallback.key] : [])
  }

  function handleModeClick(modeRecord) {
    setModes((current) => {
      if (current.includes(modeRecord.key)) {
        return current.length === 1 ? current : current.filter((key) => key !== modeRecord.key)
      }
      return [...current, modeRecord.key]
    })
  }

  return (
    <div className="h-screen overflow-hidden bg-background text-text-primary">
      <div className="grid h-full grid-cols-[260px_minmax(0,1fr)]">
        <Sidebar />
        <div className="flex min-w-0 flex-col">
          <Topbar />
          <main className="min-h-0 flex-1 overflow-y-auto p-4">
            <div className="mx-auto grid min-h-[640px] max-w-[1240px] grid-cols-[minmax(360px,0.82fr)_minmax(520px,1.18fr)] gap-4">
              <section className="flex min-w-0 flex-col gap-3">
                <Card title="Data Source">
                  <div className="rounded-lg border border-border bg-subtle px-3 py-2">
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <Icon className="text-lg text-primary">dns</Icon>
                      Database
                    </div>
                    <p className="mt-1 text-xs leading-5 text-text-muted">Built-in experimental datasets are the calibration source.</p>
                  </div>
                  <label className="mt-3 block text-xs font-bold uppercase text-text-muted">
                    Publication / source
                    <select className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm normal-case text-text-primary" value={author} onChange={(event) => handleAuthorChange(event.target.value)}>
                      {datasets.map((item) => (
                        <option key={item.author} value={item.author}>
                          {item.author}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="mt-3">
                    <Label>Stress Type</Label>
                    <div className="rounded-lg border border-border bg-subtle px-3 py-2 text-sm font-semibold text-text-primary">
                      {preview.metadata?.stressType ?? "From dataset"}
                      <span className="ml-2 align-middle text-xs font-normal text-text-muted">read from experimental data</span>
                    </div>
                  </div>
                </Card>

                <Card title="Fitting Data Sets">
                  <p className="mb-2 text-xs leading-5 text-text-muted">
                    Select one or more experimental sets to include in the calibration objective. Each database entry is shown separately.
                  </p>
                  <div className="flex max-h-72 flex-col gap-2 overflow-y-auto pr-1">
                    {authorModes.map((option) => (
                      <ModeButton
                        key={option.key}
                        option={option}
                        meta={modeOptions.find((item) => item.family === option.family) ?? modeOptions[0]}
                        active={modes.includes(option.key)}
                        onClick={() => handleModeClick(option)}
                      />
                    ))}
                  </div>
                </Card>

                <Card title="Dataset Metadata" className="flex-1">
                  <Metadata preview={preview} apiState={apiState} />
                </Card>
              </section>

              <section className="min-w-0">
                <PlotCard preview={preview} mode={primaryModeMeta} />
              </section>
            </div>
          </main>
          <BottomBar rows={preview.metadata?.rows ?? 0} />
        </div>
      </div>
    </div>
  )
}

function Sidebar() {
  const items = [
    ["science", "Experimental Data", "active"],
    ["schema", "Model Architecture", "ready"],
    ["tune", "Optimization", "locked"],
    ["analytics", "Prediction", "locked"],
  ]
  return (
    <aside className="flex h-full flex-col border-r border-border bg-surface px-3 py-4">
      <div className="px-2 pb-6">
        <h1 className="text-[15px] font-semibold">Calibration Shell</h1>
        <p className="mt-0.5 text-xs text-text-muted">Precision Workflow</p>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {items.map(([icon, label, state]) => (
          <button
            key={label}
            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition ${
              state === "active"
                ? "bg-selection-bg font-semibold text-primary"
                : state === "locked"
                  ? "text-text-disabled"
                  : "text-text-primary hover:bg-subtle"
            }`}
          >
            <Icon className="text-lg">{icon}</Icon>
            <span className="flex-1">{label}</span>
            {state === "active" && <Icon className="text-base">chevron_right</Icon>}
          </button>
        ))}
      </nav>
      <div className="border-t border-border pt-3">
        {[
          ["menu_book", "Documentation"],
          ["settings", "Settings"],
        ].map(([icon, label]) => (
          <button key={label} className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-text-primary hover:bg-subtle">
            <Icon className="text-lg">{icon}</Icon>
            {label}
          </button>
        ))}
      </div>
    </aside>
  )
}

function Topbar() {
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-surface px-4">
      <div>
        <h2 className="text-lg font-semibold">Calibration for Hyperelasticity</h2>
        <p className="text-xs text-text-muted">Project: Experimental Data Workspace</p>
      </div>
      <div className="flex items-center gap-3">
        <label className="relative">
          <Icon className="absolute left-2 top-1/2 -translate-y-1/2 text-lg text-text-muted">search</Icon>
          <input className="h-10 w-56 rounded-lg border border-border-strong bg-surface pl-8 pr-3 text-sm outline-none focus:border-primary" placeholder="Search..." />
        </label>
        <button className="grid h-9 w-9 place-items-center rounded-lg border border-border hover:bg-subtle">
          <Icon className="text-lg">settings</Icon>
        </button>
        <button className="grid h-9 w-9 place-items-center rounded-lg border border-border hover:bg-subtle">
          <Icon className="text-lg">help</Icon>
        </button>
      </div>
    </header>
  )
}

function Card({ title, children, className = "" }) {
  return (
    <div className={`rounded-lg border border-border bg-surface p-3 shadow-panel ${className}`}>
      <h3 className="mb-3 text-[15px] font-semibold">{title}</h3>
      {children}
    </div>
  )
}

function Label({ children }) {
  return <div className="mb-1 text-xs font-bold uppercase text-text-muted">{children}</div>
}

function ModeButton({ option, meta, active, onClick }) {
  return (
    <button
      className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-left ${
        active
          ? "border-primary bg-selection-bg"
          : "border-border-strong bg-surface hover:bg-subtle"
      }`}
      onClick={onClick}
    >
      <span className={`h-3 w-3 rounded-full ${meta.dot}`} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold">{option.label}</span>
        <span className="mt-0.5 block truncate text-xs font-normal text-text-muted">
          {option.key} · {option.points} pts · {option.stressType}
        </span>
      </span>
      <span className={`grid h-5 w-5 place-items-center rounded border ${active ? "border-primary bg-primary text-white" : "border-border-strong bg-surface"}`}>
        {active && <Icon className="text-sm">check_circle</Icon>}
      </span>
    </button>
  )
}

function Metadata({ preview, apiState }) {
  const metadata = preview.metadata ?? {}
  return (
    <dl className="grid grid-cols-2 gap-2 text-sm">
      <MetaItem label="Rows parsed" value={metadata.rows ?? 0} />
      <MetaItem label="API" value={apiState} />
      <MetaItem label="Selected sets" value={metadata.setCount ?? 0} />
      <MetaItem label="Stress type" value={metadata.stressType ?? "-"} />
      <MetaItem label="Source" value="Database" />
      <MetaItem label="Reference" value={metadata.source ?? "-"} />
      <div className="col-span-2 rounded-lg border border-border bg-subtle p-2">
        <dt className="text-[11px] font-bold uppercase text-text-muted">Loading modes</dt>
        <dd className="mt-1 text-sm font-semibold">{metadata.selectedMode ?? "-"}</dd>
      </div>
    </dl>
  )
}

function MetaItem({ label, value }) {
  return (
    <div className="rounded-lg border border-border bg-subtle p-2">
      <dt className="text-[11px] font-bold uppercase text-text-muted">{label}</dt>
      <dd className="mt-1 truncate text-sm font-semibold">{value}</dd>
    </div>
  )
}

function PlotCard({ preview, mode }) {
  return (
    <div className="flex h-full min-h-[640px] flex-col overflow-hidden rounded-lg border border-border bg-surface shadow-panel">
      <div className="flex items-center justify-between border-b border-border bg-subtle px-4 py-3">
        <h3 className="flex items-center gap-2 text-[15px] font-semibold">
          <Icon className="text-lg">monitoring</Icon>
          Preview: Experimental Stress-Stretch
        </h3>
        <div className="flex gap-1">
          {["zoom_in", "restart_alt", "save"].map((icon) => (
            <button key={icon} className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-surface">
              <Icon className="text-lg">{icon}</Icon>
            </button>
          ))}
        </div>
      </div>
      <div className="min-h-0 flex-1 p-5">
        <ScientificChart preview={preview} xLabel={preview.axes?.x} yLabel={preview.axes?.y} mode={mode} />
      </div>
      <div className="flex items-center justify-between border-t border-border px-4 py-3 text-sm">
        <span className="text-text-muted">Data sampled from {preview.metadata?.source ?? "current selection"}</span>
        <button className="rounded-lg border border-border-strong bg-surface px-3 py-1.5 font-semibold hover:bg-subtle">Save Plot</button>
      </div>
    </div>
  )
}

function ScientificChart({ preview, xLabel, yLabel, mode }) {
  const series = preview.series?.length
    ? preview.series
    : [{ modeFamily: mode.family, modeLabel: mode.label, points: preview.points ?? [] }]
  const points = series.flatMap((item) => item.points ?? [])
  const width = 720
  const height = 460
  const pad = { top: 34, right: 28, bottom: 58, left: 78 }
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs, 0)
  const maxX = Math.max(...xs, 1)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys, 1)
  const scaleX = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (width - pad.left - pad.right)
  const scaleY = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (height - pad.top - pad.bottom)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  const legendItems = series.slice(0, 5)

  return (
    <div className="relative h-full w-full">
      <svg className="h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Experimental stress stretch chart">
        <rect x="0" y="0" width={width} height={height} fill="#FFFFFF" />
        {ticks.map((tick) => {
          const x = pad.left + tick * (width - pad.left - pad.right)
          const y = pad.top + tick * (height - pad.top - pad.bottom)
          return (
            <g key={tick}>
              <line x1={x} y1={pad.top} x2={x} y2={height - pad.bottom} stroke="#E5E5EA" strokeDasharray="4 4" />
              <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke="#E5E5EA" strokeDasharray="4 4" />
            </g>
          )
        })}
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} stroke="#D1D1D6" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="#D1D1D6" />
        {series.map((item, seriesIndex) => {
          const meta = modeOptions.find((option) => option.family === item.modeFamily) ?? modeOptions[seriesIndex % modeOptions.length]
          const color = colorForSeries(meta.family, seriesIndex)
          const seriesPath = (item.points ?? [])
            .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
            .join(" ")
          return (
            <g key={item.mode}>
              {seriesPath && <path d={seriesPath} fill="none" stroke={color} strokeWidth="2" opacity="0.42" />}
              {(item.points ?? []).map((point, index) => (
                <circle key={`${item.mode}-${point.x}-${point.y}-${index}`} cx={scaleX(point.x)} cy={scaleY(point.y)} r="4" fill={color} stroke="#FFFFFF" strokeWidth="1.5" />
              ))}
            </g>
          )
        })}
        <text x={width / 2} y={height - 16} textAnchor="middle" fontSize="12" fill="#6E6E73">
          {xLabel}
        </text>
        <text x="18" y={height / 2} textAnchor="middle" fontSize="12" fill="#6E6E73" transform={`rotate(-90 18 ${height / 2})`}>
          {yLabel}
        </text>
      </svg>
      <div className="absolute right-4 top-4 max-w-[280px] rounded-md border border-border-strong bg-white/95 px-2 py-1.5 text-xs shadow-panel">
        <div className="mb-1 font-semibold text-text-primary">Selected data</div>
        <div className="flex max-h-28 flex-col gap-1 overflow-hidden">
          {legendItems.map((item, index) => (
            <div key={item.mode ?? item.modeLabel} className="flex min-w-0 items-center gap-2">
              <span className="h-2 w-4 shrink-0 rounded-full" style={{ backgroundColor: colorForSeries(item.modeFamily, index) }} />
              <span className="truncate text-text-muted">{item.modeLabel ?? item.mode ?? "Experimental set"}</span>
            </div>
          ))}
        </div>
        {series.length > legendItems.length && <div className="mt-1 text-text-muted">+{series.length - legendItems.length} more</div>}
      </div>
    </div>
  )
}

function colorForSeries(family, index = 0) {
  const base = {
    CSS: "#0891B2",
    UT: "#2563EB",
    PS: "#059669",
    ET: "#EF4444",
    SS: "#0F766E",
    BT: "#7C3AED",
  }[family] ?? "#007AFF"
  const palette = ["#2563EB", "#0891B2", "#059669", "#DC2626", "#7C3AED", "#EA580C", "#475569"]
  return index === 0 ? base : palette[index % palette.length]
}

function BottomBar({ rows }) {
  return (
    <footer className="flex h-16 shrink-0 items-center justify-between border-t border-border bg-surface px-4">
      <button className="flex items-center gap-1 rounded-lg border border-border-strong px-4 py-2 text-sm font-semibold hover:bg-subtle">
        <Icon className="text-lg">arrow_back</Icon>
        Back
      </button>
      <div className="flex items-center gap-4">
        <span className="text-sm text-text-muted">Status: Ready · {rows} rows parsed</span>
        <button className="flex items-center gap-1 rounded-lg bg-primary px-5 py-2 text-sm font-semibold text-white shadow-panel hover:bg-primary-hover">
          Next Step
          <Icon className="text-lg">arrow_forward</Icon>
        </button>
      </div>
    </footer>
  )
}

createRoot(document.getElementById("root")).render(<App />)
