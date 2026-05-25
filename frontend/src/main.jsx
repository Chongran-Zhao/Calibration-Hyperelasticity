import React, { useEffect, useMemo, useState } from "react"
import { createRoot } from "react-dom/client"
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  Info,
  Layers,
  LineChart,
  Plus,
  Power,
  RotateCcw,
  Save,
  Search,
  Settings,
  SlidersHorizontal,
  Trash2,
  ZoomIn,
} from "lucide-react"
import katex from "katex"
import "katex/dist/katex.min.css"
import "./styles.css"

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"
const defaultBranches = [
  { id: "spring-1", name: "Spring 1", modelKey: "ZhanNonGaussian", enabled: true },
  { id: "spring-2", name: "Spring 2", modelKey: "NeoHookean", enabled: true },
  { id: "spring-3", name: "Spring 3", modelKey: "Ogden_2term", enabled: false },
]

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
  arrow_downward: ArrowDown,
  arrow_forward: ArrowRight,
  arrow_upward: ArrowUp,
  check_circle: CheckCircle2,
  chevron_right: ChevronRight,
  dns: Database,
  help: CircleHelp,
  info: Info,
  menu_book: BookOpen,
  monitoring: Activity,
  add: Plus,
  power_settings_new: Power,
  restart_alt: RotateCcw,
  save: Save,
  schema: Layers,
  science: FlaskConical,
  search: Search,
  settings: Settings,
  delete: Trash2,
  tune: SlidersHorizontal,
  zoom_in: ZoomIn,
}

function Icon({ children, className = "" }) {
  const Component = iconMap[children] ?? Info
  return <Component aria-hidden="true" className={`inline-block h-[1em] w-[1em] shrink-0 ${className}`} />
}

function App() {
  const [activeStep, setActiveStep] = useState("experimental")
  const [datasets, setDatasets] = useState([])
  const [author, setAuthor] = useState("")
  const [modes, setModes] = useState([])
  const [modelCatalog, setModelCatalog] = useState([])
  const [branches, setBranches] = useState(defaultBranches)
  const [selectedBranchId, setSelectedBranchId] = useState(defaultBranches[0].id)
  const [parameterOverrides, setParameterOverrides] = useState({})
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
    Promise.all([fetch(`${API_BASE}/api/datasets`), fetch(`${API_BASE}/api/models`)])
      .then(async ([datasetRes, modelRes]) => {
        const datasetData = await datasetRes.json()
        const modelData = await modelRes.json()
        const authors = datasetData.authors ?? []
        const models = modelData.models ?? []
        setDatasets(authors)
        setModelCatalog(models)
        const fallbackModel = models.find((item) => item.key === "ZhanNonGaussian")?.key ?? models[0]?.key ?? ""
        setBranches((current) => current.map((branch) => ({
          ...branch,
          modelKey: models.some((model) => model.key === branch.modelKey) ? branch.modelKey : fallbackModel,
        })))
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
  const selectedBranch = useMemo(
    () => branches.find((branch) => branch.id === selectedBranchId) ?? branches[0],
    [branches, selectedBranchId],
  )
  const selectedModel = useMemo(
    () => modelCatalog.find((item) => item.key === selectedBranch?.modelKey) ?? modelCatalog[0],
    [modelCatalog, selectedBranch],
  )
  const selectedParameterValues = useMemo(() => {
    if (!selectedModel || !selectedBranch) return {}
    const overrides = parameterOverrides[selectedBranch.id] ?? {}
    return Object.fromEntries(
      selectedModel.parameters.map((param) => [
        param.name,
        overrides[param.name] ?? String(param.initial ?? ""),
      ]),
    )
  }, [parameterOverrides, selectedBranch, selectedModel])

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

  function handleParameterChange(paramName, value) {
    if (!selectedBranch) return
    setParameterOverrides((current) => ({
      ...current,
      [selectedBranch.id]: {
        ...(current[selectedBranch.id] ?? {}),
        [paramName]: value,
      },
    }))
  }

  function resetSelectedParameters() {
    if (!selectedBranch) return
    setParameterOverrides((current) => {
      const next = { ...current }
      delete next[selectedBranch.id]
      return next
    })
  }

  function updateBranch(branchId, patch) {
    setBranches((current) => current.map((branch) => branch.id === branchId ? { ...branch, ...patch } : branch))
  }

  function addBranch() {
    const modelKey = modelCatalog.find((model) => model.key === "NeoHookean")?.key ?? modelCatalog[0]?.key ?? ""
    const nextIndex = branches.length + 1
    const branch = {
      id: `spring-${Date.now()}`,
      name: `Spring ${nextIndex}`,
      modelKey,
      enabled: true,
    }
    setBranches((current) => [...current, branch])
    setSelectedBranchId(branch.id)
  }

  function removeBranch(branchId) {
    setBranches((current) => {
      if (current.length === 1) return current
      const next = current.filter((branch) => branch.id !== branchId)
      if (selectedBranchId === branchId) {
        setSelectedBranchId(next[0].id)
      }
      return next
    })
    setParameterOverrides((current) => {
      const next = { ...current }
      delete next[branchId]
      return next
    })
  }

  function moveBranch(branchId, direction) {
    setBranches((current) => {
      const index = current.findIndex((branch) => branch.id === branchId)
      const target = index + direction
      if (index < 0 || target < 0 || target >= current.length) return current
      const next = [...current]
      const [item] = next.splice(index, 1)
      next.splice(target, 0, item)
      return next
    })
  }

  return (
    <div className="h-screen overflow-hidden bg-background text-text-primary">
      <div className="grid h-full grid-cols-[260px_minmax(0,1fr)]">
        <Sidebar activeStep={activeStep} onStepChange={setActiveStep} />
        <div className="flex min-w-0 flex-col">
          <Topbar activeStep={activeStep} />
          <main className="min-h-0 flex-1 overflow-y-auto p-4">
            {activeStep === "experimental" ? (
              <ExperimentalDataPage
                datasets={datasets}
                author={author}
                authorModes={authorModes}
                modes={modes}
                preview={preview}
                apiState={apiState}
                primaryModeMeta={primaryModeMeta}
                onAuthorChange={handleAuthorChange}
                onModeClick={handleModeClick}
              />
            ) : (
              <ModelArchitecturePage
                models={modelCatalog}
                branches={branches}
                selectedBranch={selectedBranch}
                selectedModel={selectedModel}
                parameterValues={selectedParameterValues}
                onAddBranch={addBranch}
                onRemoveBranch={removeBranch}
                onSelectBranch={setSelectedBranchId}
                onMoveBranch={moveBranch}
                onUpdateBranch={updateBranch}
                onParameterChange={handleParameterChange}
                onResetParameters={resetSelectedParameters}
                selectedDataCount={modes.length}
              />
            )}
          </main>
          <BottomBar
            activeStep={activeStep}
            rows={preview.metadata?.rows ?? 0}
            selectedBranch={selectedBranch}
            selectedModel={selectedModel}
            onNext={() => setActiveStep(activeStep === "experimental" ? "models" : "experimental")}
          />
        </div>
      </div>
    </div>
  )
}

function ExperimentalDataPage({
  datasets,
  author,
  authorModes,
  modes,
  preview,
  apiState,
  primaryModeMeta,
  onAuthorChange,
  onModeClick,
}) {
  return (
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
            <select className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm normal-case text-text-primary" value={author} onChange={(event) => onAuthorChange(event.target.value)}>
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
                onClick={() => onModeClick(option)}
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
  )
}

function ModelArchitecturePage({
  models,
  branches,
  selectedBranch,
  selectedModel,
  parameterValues,
  onAddBranch,
  onRemoveBranch,
  onSelectBranch,
  onMoveBranch,
  onUpdateBranch,
  onParameterChange,
  onResetParameters,
  selectedDataCount,
}) {
  const modelByKey = useMemo(
    () => Object.fromEntries(models.map((model) => [model.key, model])),
    [models],
  )
  const activeBranches = branches.filter((branch) => branch.enabled)

  return (
    <div className="mx-auto flex min-h-[640px] max-w-[1240px] flex-col gap-4">
      <ArchitectureSummary branches={branches} activeBranches={activeBranches} selectedDataCount={selectedDataCount} selectedBranch={selectedBranch} />
      <section className="grid min-h-[560px] gap-4 xl:grid-cols-[minmax(300px,0.78fr)_minmax(420px,1.2fr)_minmax(360px,0.95fr)]">
        <Card title="Parallel Branches">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase text-text-muted">{activeBranches.length} active · {branches.length} total</p>
            <button className="flex shrink-0 items-center gap-1 rounded-lg bg-primary px-3 py-1.5 text-sm font-semibold text-white shadow-panel hover:bg-primary-hover" onClick={onAddBranch}>
              <Icon className="text-base">add</Icon>
              Add
            </button>
          </div>
          <div className="flex max-h-[500px] flex-col gap-2 overflow-y-auto pr-1">
            {branches.map((branch, index) => (
              <BranchCard
                key={branch.id}
                branch={branch}
                index={index}
                total={branches.length}
                model={modelByKey[branch.modelKey]}
                active={branch.id === selectedBranch?.id}
                onSelect={() => onSelectBranch(branch.id)}
                onToggle={() => onUpdateBranch(branch.id, { enabled: !branch.enabled })}
                onMove={(direction) => onMoveBranch(branch.id, direction)}
                onRemove={() => onRemoveBranch(branch.id)}
              />
            ))}
          </div>
        </Card>

        <Card title="Parallel Spring Architecture">
          <div className="rounded-lg border border-border bg-subtle p-3">
            <ArchitectureVisualizer branches={branches} selectedBranchId={selectedBranch?.id} modelByKey={modelByKey} onSelectBranch={onSelectBranch} />
          </div>
          <div className="mt-3 rounded-lg border border-border bg-subtle p-3">
            <Label>Composition</Label>
            <div className="text-sm leading-6 text-text-primary">
              Total response: <span className="font-mono">P = Σ Pᵢ(F)</span>
              <span className="ml-2 text-text-muted">with common deformation gradient F across active branches.</span>
            </div>
          </div>
        </Card>

        <section className="flex min-w-0 flex-col gap-3">
          <Card title="Selected Branch">
            {selectedBranch ? (
              <div className="space-y-3">
                <label className="block">
                  <Label>Branch Name</Label>
                  <input className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={selectedBranch.name} onChange={(event) => onUpdateBranch(selectedBranch.id, { name: event.target.value })} />
                </label>
                <label className="block">
                  <Label>Constitutive Model</Label>
                  <select className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={selectedBranch.modelKey} onChange={(event) => onUpdateBranch(selectedBranch.id, { modelKey: event.target.value })}>
                    {models.map((model) => (
                      <option key={model.key} value={model.key}>{model.name}</option>
                    ))}
                  </select>
                </label>
                <button className={`flex h-9 w-full items-center justify-center gap-1 rounded-lg border px-3 text-sm font-semibold ${selectedBranch.enabled ? "border-primary bg-selection-bg text-primary" : "border-border-strong bg-surface text-text-muted"}`} onClick={() => onUpdateBranch(selectedBranch.id, { enabled: !selectedBranch.enabled })}>
                  <Icon className="text-base">power_settings_new</Icon>
                  {selectedBranch.enabled ? "Enabled in Architecture" : "Disabled"}
                </button>
                {selectedModel && (
                  <>
                    <div className="rounded-lg border border-border bg-subtle p-3">
                      <h3 className="text-sm font-semibold">{selectedModel.name}</h3>
                      <p className="mt-1 text-xs text-text-muted">{typeLabel(selectedModel.type)} · {selectedModel.category}</p>
                    </div>
                    <FormulaBlock label="Branch Energy Density" value={selectedModel.formula} />
                    {selectedModel.strainFormula && <FormulaBlock label="Generalized Strain" value={selectedModel.strainFormula} />}
                  </>
                )}
              </div>
            ) : (
              <p className="text-sm text-text-muted">No branch selected.</p>
            )}
          </Card>
          <Card title="Branch Parameters" className="flex-1">
            <ParameterTable parameters={selectedModel?.parameters ?? []} values={parameterValues} onChange={onParameterChange} onReset={onResetParameters} />
          </Card>
        </section>
      </section>
    </div>
  )
}

function ArchitectureSummary({ branches, activeBranches, selectedDataCount, selectedBranch }) {
  const items = [
    ["Total branches", branches.length],
    ["Active branches", activeBranches.length],
    ["Experimental sets", selectedDataCount],
    ["Editing branch", selectedBranch?.name ?? "-"],
  ]
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-lg border border-border bg-surface p-3 shadow-panel">
          <div className="text-[11px] font-bold uppercase text-text-muted">{label}</div>
          <div className="mt-1 text-lg font-semibold">{value}</div>
        </div>
      ))}
    </div>
  )
}

function BranchCard({ branch, index, total, model, active, onSelect, onToggle, onMove, onRemove }) {
  return (
    <div
      className={`rounded-lg border p-3 text-left transition ${active ? "border-primary bg-selection-bg" : branch.enabled ? "border-border-strong bg-surface hover:bg-subtle" : "border-border bg-subtle opacity-70"}`}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          onSelect()
        }
      }}
      role="button"
      tabIndex={0}
    >
      <div className="flex items-start gap-2">
        <span className={`mt-1 h-3 w-3 shrink-0 rounded-full ${branch.enabled ? "bg-primary" : "bg-text-disabled"}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-sm font-semibold">{branch.name}</span>
            <span className={`rounded border px-1.5 py-0.5 text-[11px] font-semibold ${branch.enabled ? "border-primary bg-selection-bg text-primary" : "border-border-strong bg-surface text-text-muted"}`}>
              {branch.enabled ? "active" : "off"}
            </span>
          </div>
          <div className="mt-1 truncate text-xs text-text-muted">{model?.name ?? branch.modelKey}</div>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-1">
        <IconButton label={branch.enabled ? "Disable" : "Enable"} onClick={onToggle}>power_settings_new</IconButton>
        <IconButton label="Move up" disabled={index === 0} onClick={() => onMove(-1)}>arrow_upward</IconButton>
        <IconButton label="Move down" disabled={index === total - 1} onClick={() => onMove(1)}>arrow_downward</IconButton>
        <IconButton label="Remove" disabled={total === 1} onClick={onRemove}>delete</IconButton>
      </div>
    </div>
  )
}

function IconButton({ children, label, disabled = false, onClick }) {
  return (
    <button
      className="grid h-8 place-items-center rounded-md border border-border bg-surface text-text-muted hover:bg-subtle disabled:opacity-40"
      disabled={disabled}
      title={label}
      onClick={(event) => {
        event.stopPropagation()
        onClick?.()
      }}
    >
      <Icon className="text-base">{children}</Icon>
    </button>
  )
}

function ArchitectureVisualizer({ branches, selectedBranchId, modelByKey, onSelectBranch }) {
  const width = 720
  const height = Math.max(300, 92 + branches.length * 78)
  const railLeft = 86
  const railRight = width - 86
  const top = 64
  const rowGap = 72
  return (
    <div className="h-[420px] overflow-auto">
      <svg className="min-h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Parallel spring architecture">
        <rect x="0" y="0" width={width} height={height} rx="10" fill="#FFFFFF" />
        <line x1={railLeft} y1={top - 24} x2={railLeft} y2={top + (branches.length - 1) * rowGap + 24} stroke="#717786" strokeWidth="4" strokeLinecap="round" />
        <line x1={railRight} y1={top - 24} x2={railRight} y2={top + (branches.length - 1) * rowGap + 24} stroke="#717786" strokeWidth="4" strokeLinecap="round" />
        <text x={railLeft} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#6E6E73">F input</text>
        <text x={railRight} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#6E6E73">P total</text>
        {branches.map((branch, index) => {
          const y = top + index * rowGap
          const selected = branch.id === selectedBranchId
          const color = selected ? "#007AFF" : branch.enabled ? "#414755" : "#C1C6D7"
          const opacity = branch.enabled ? 1 : 0.45
          const springStart = railLeft + 70
          const springEnd = railRight - 70
          const path = springPath(springStart, springEnd, y)
          return (
            <g key={branch.id} opacity={opacity} onClick={() => onSelectBranch(branch.id)} className="cursor-pointer">
              <line x1={railLeft} y1={y} x2={springStart} y2={y} stroke={color} strokeWidth={selected ? "3" : "2"} />
              <path d={path} fill="none" stroke={color} strokeWidth={selected ? "3" : "2"} strokeLinejoin="round" />
              <line x1={springEnd} y1={y} x2={railRight} y2={y} stroke={color} strokeWidth={selected ? "3" : "2"} />
              <circle cx={railLeft} cy={y} r="7" fill="#FFFFFF" stroke={color} strokeWidth="2" />
              <circle cx={railRight} cy={y} r="7" fill="#FFFFFF" stroke={color} strokeWidth="2" />
              <rect x={springStart + 112} y={y - 25} width="210" height="50" rx="8" fill={selected ? "#EAF3FF" : "#F9F9FB"} stroke={selected ? "#007AFF" : "#E5E5EA"} />
              <text x={springStart + 126} y={y - 5} fontSize="12" fontWeight="700" fill="#1C1C1E">{branch.name}</text>
              <text x={springStart + 126} y={y + 13} fontSize="11" fill="#6E6E73">{modelByKey[branch.modelKey]?.name ?? branch.modelKey}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function springPath(startX, endX, y) {
  const coils = 8
  const amp = 14
  const step = (endX - startX) / (coils * 2)
  const points = [`M ${startX} ${y}`]
  for (let index = 1; index < coils * 2; index += 1) {
    points.push(`L ${startX + step * index} ${y + (index % 2 ? -amp : amp)}`)
  }
  points.push(`L ${endX} ${y}`)
  return points.join(" ")
}

function Sidebar({ activeStep, onStepChange }) {
  const items = [
    ["experimental", "science", "Experimental Data", "ready"],
    ["models", "schema", "Model Architecture", "ready"],
    ["optimization", "tune", "Optimization", "locked"],
    ["prediction", "analytics", "Prediction", "locked"],
  ]
  return (
    <aside className="flex h-full flex-col border-r border-border bg-surface px-3 py-4">
      <div className="px-2 pb-6">
        <h1 className="text-[15px] font-semibold">Calibration Shell</h1>
        <p className="mt-0.5 text-xs text-text-muted">Precision Workflow</p>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {items.map(([key, icon, label, state]) => {
          const active = activeStep === key
          return (
          <button
            key={label}
            onClick={() => state !== "locked" && onStepChange(key)}
            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition ${
              active
                ? "bg-selection-bg font-semibold text-primary"
                : state === "locked"
                  ? "text-text-disabled"
                  : "text-text-primary hover:bg-subtle"
            }`}
          >
            <Icon className="text-lg">{icon}</Icon>
            <span className="flex-1">{label}</span>
            {active && <Icon className="text-base">chevron_right</Icon>}
          </button>
          )
        })}
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

function Topbar({ activeStep }) {
  const title = activeStep === "models" ? "Model Architecture" : "Experimental Data Workspace"
  const subtitle = activeStep === "models" ? "Constitutive model selection and parameter setup" : "Project: Experimental Data Workspace"
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-surface px-4">
      <div>
        <h2 className="text-lg font-semibold">Calibration for Hyperelasticity</h2>
        <p className="text-xs text-text-muted">{subtitle} · {title}</p>
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

function FormulaBlock({ label, value }) {
  const rendered = useMemo(() => renderFormula(value), [value])
  return (
    <div>
      <Label>{label}</Label>
      {rendered ? (
        <div className="formula-block min-h-20 overflow-x-auto rounded-lg border border-border bg-subtle px-4 py-4 text-text-primary" dangerouslySetInnerHTML={{ __html: rendered }} />
      ) : (
        <div className="rounded-lg border border-border bg-subtle px-3 py-2 text-sm text-text-muted">-</div>
      )}
    </div>
  )
}

function ParameterTable({ parameters, values, onChange, onReset }) {
  if (!parameters.length) {
    return <p className="text-sm text-text-muted">This model does not expose editable parameters.</p>
  }
  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-lg border border-border">
        <div className="grid grid-cols-[1fr_1.2fr_1fr_1fr] border-b border-border bg-subtle px-3 py-2 text-xs font-bold uppercase text-text-muted">
          <span>Name</span>
          <span>Initial</span>
          <span>Lower</span>
          <span>Upper</span>
        </div>
        {parameters.map((param) => (
          <div key={param.name} className="grid grid-cols-[1fr_1.2fr_1fr_1fr] items-center gap-2 border-b border-border px-3 py-2 text-sm last:border-b-0">
            <span className="font-semibold">
              <LatexInline value={parameterSymbol(param.name)} fallback={param.name} />
            </span>
            <input
              className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary"
              type="number"
              step="any"
              value={values[param.name] ?? ""}
              onChange={(event) => onChange(param.name, event.target.value)}
            />
            <span>{formatBound(param.bounds?.[0])}</span>
            <span>{formatBound(param.bounds?.[1])}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-between">
        <p className="text-xs leading-5 text-text-muted">Initial values are editable and kept for the selected model while you move through the workflow.</p>
        <button className="rounded-lg border border-border-strong bg-surface px-3 py-1.5 text-sm font-semibold hover:bg-subtle" onClick={onReset}>
          Reset
        </button>
      </div>
    </div>
  )
}

function LatexInline({ value, fallback }) {
  const rendered = useMemo(() => renderFormula(value, false), [value])
  return rendered ? <span dangerouslySetInnerHTML={{ __html: rendered }} /> : fallback
}

function parameterSymbol(name) {
  const indexed = name.match(/^([A-Za-z]+)(\d+)$/)
  if (indexed) {
    const [, base, index] = indexed
    return `${parameterSymbol(base)}_{${index}}`
  }
  return {
    alpha: "\\alpha",
    beta: "\\beta",
    gamma: "\\gamma",
    lambda: "\\lambda",
    mu: "\\mu",
  }[name] ?? name
}

function renderFormula(value, displayMode = true) {
  if (!value) return ""
  try {
    return katex.renderToString(value, {
      displayMode,
      throwOnError: false,
      strict: false,
      trust: false,
    })
  } catch {
    return ""
  }
}

function typeLabel(type) {
  return {
    custom: "Custom Stress",
    invariant_based: "Invariant Based",
    stretch_based: "Stretch Based",
  }[type] ?? type
}

function formatBound(value) {
  return value === null || value === undefined ? "free" : Number(value).toPrecision(4)
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

function BottomBar({ activeStep, rows, selectedBranch, selectedModel, onNext }) {
  const status = activeStep === "models"
    ? `Editing ${selectedBranch?.name ?? "branch"} · ${selectedModel?.name ?? "-"}`
    : `Status: Ready · ${rows} rows parsed`
  const nextLabel = activeStep === "models" ? "Back to Data" : "Next Step"
  return (
    <footer className="flex h-16 shrink-0 items-center justify-between border-t border-border bg-surface px-4">
      <button className="flex items-center gap-1 rounded-lg border border-border-strong px-4 py-2 text-sm font-semibold hover:bg-subtle">
        <Icon className="text-lg">arrow_back</Icon>
        Back
      </button>
      <div className="flex items-center gap-4">
        <span className="text-sm text-text-muted">{status}</span>
        <button className="flex items-center gap-1 rounded-lg bg-primary px-5 py-2 text-sm font-semibold text-white shadow-panel hover:bg-primary-hover" onClick={onNext}>
          {nextLabel}
          <Icon className="text-lg">arrow_forward</Icon>
        </button>
      </div>
    </footer>
  )
}

createRoot(document.getElementById("root")).render(<App />)
