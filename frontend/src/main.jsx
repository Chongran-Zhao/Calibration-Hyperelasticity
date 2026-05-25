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
  Plus,
  Power,
  Play,
  RotateCcw,
  Save,
  Search,
  Settings,
  SlidersHorizontal,
  Square,
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
  { id: "spring-3", name: "Spring 3", modelKey: "Ogden", modelConfig: { termCount: 2 }, enabled: false },
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
  arrow_forward: ArrowRight,
  check_circle: CheckCircle2,
  chevron_right: ChevronRight,
  dns: Database,
  help: CircleHelp,
  info: Info,
  menu_book: BookOpen,
  monitoring: Activity,
  add: Plus,
  play_arrow: Play,
  power_settings_new: Power,
  restart_alt: RotateCcw,
  save: Save,
  schema: Layers,
  science: FlaskConical,
  search: Search,
  settings: Settings,
  stop: Square,
  delete: Trash2,
  tune: SlidersHorizontal,
  zoom_in: ZoomIn,
}

const solverMethods = ["L-BFGS-B", "trust-constr", "trf", "dogbox", "lm"]

const defaultSolverSettings = {
  method: "L-BFGS-B",
  maxIter: 500,
  r2Target: 0.995,
  absTol: 0.000001,
  relTol: 0.0001,
  maxLoss: 0.05,
}

const initialOptimizationState = {
  status: "Ready",
  running: false,
  progress: 0,
  initialLoss: 0.2468,
  currentLoss: 0.2468,
  r2: 0.0,
  iterations: 0,
  log: ["Ready to calibrate selected model and datasets."],
}

const defaultPredictionSettings = {
  source: "lastOptimization",
  mode: "UT",
  stretchMin: 1,
  stretchMax: 3.5,
  samples: 72,
  stressOutput: "First PK",
  manualEdits: false,
}

function Icon({ children, className = "" }) {
  const Component = iconMap[children] ?? Info
  return <Component aria-hidden="true" className={`inline-block h-[1em] w-[1em] shrink-0 ${className}`} />
}

function defaultConfigForModel(model) {
  const kind = model?.configurable?.kind
  if (kind === "ogden") {
    return { termCount: 1 }
  }
  if (kind === "hill") {
    const firstStrain = model.configurable?.strains?.[0]?.key ?? ""
    return { terms: [{ strain: firstStrain }] }
  }
  return {}
}

function normalizeConfigForModel(model, config = {}) {
  const kind = model?.configurable?.kind
  if (kind === "ogden") {
    const min = model.configurable?.minTerms ?? 1
    const max = model.configurable?.maxTerms ?? 5
    const termCount = Math.min(max, Math.max(min, Number(config.termCount ?? 1) || 1))
    return { termCount }
  }
  if (kind === "hill") {
    const min = model.configurable?.minTerms ?? 1
    const max = model.configurable?.maxTerms ?? 5
    const strains = model.configurable?.strains ?? []
    const fallback = strains[0]?.key ?? ""
    const allowed = new Set(strains.map((strain) => strain.key))
    const currentTerms = Array.isArray(config.terms) ? config.terms : []
    const terms = currentTerms.length ? currentTerms : [{ strain: fallback }]
    return {
      terms: terms
        .slice(0, max)
        .map((term) => ({ strain: allowed.has(term.strain) ? term.strain : fallback }))
        .concat(Array.from({ length: Math.max(0, min - terms.length) }, () => ({ strain: fallback }))),
    }
  }
  return {}
}

function buildConfiguredModel(model, config = {}) {
  if (!model) return null
  const kind = model.configurable?.kind
  if (kind === "ogden") {
    const normalized = normalizeConfigForModel(model, config)
    const termCount = normalized.termCount
    return {
      ...model,
      config: normalized,
      detail: `${termCount} term${termCount === 1 ? "" : "s"}`,
      formula: ogdenFormula(termCount),
      parameters: ogdenParameters(termCount),
      strainFormula: "",
    }
  }
  if (kind === "hill") {
    const normalized = normalizeConfigForModel(model, config)
    return {
      ...model,
      config: normalized,
      detail: `${normalized.terms.length} term${normalized.terms.length === 1 ? "" : "s"}`,
      formula: hillFormula(normalized.terms.length),
      strainFormula: hillStrainFormula(model, normalized.terms),
      parameters: hillParameters(model, normalized.terms),
    }
  }
  return model
}

function ogdenFormula(termCount) {
  return String.raw`\Psi = \sum_{k=1}^{${termCount}} \frac{\mu_k}{\alpha_k}\left(\lambda_1^{\alpha_k} + \lambda_2^{\alpha_k} + \lambda_3^{\alpha_k} - 3\right)`
}

function ogdenParameters(termCount) {
  return Array.from({ length: termCount }, (_, index) => {
    const term = index + 1
    const negativeFirstTerm = term === 1
    return [
      {
        name: `mu${term}`,
        initial: negativeFirstTerm ? -0.5 : 0.5 / term,
        bounds: negativeFirstTerm ? [null, -1e-6] : [1e-6, null],
      },
      {
        name: `alpha${term}`,
        initial: negativeFirstTerm ? -2 : 2 * term,
        bounds: negativeFirstTerm ? [null, -1e-6] : [1e-6, null],
      },
    ]
  }).flat()
}

function hillFormula(termCount) {
  return String.raw`\Psi = \sum_{k=1}^{${termCount}} \mu_k \sum_{i=1}^{3} E_k(\lambda_i)^2`
}

function hillStrainFormula(model, terms) {
  const strainByKey = Object.fromEntries((model.configurable?.strains ?? []).map((strain) => [strain.key, strain]))
  const formulas = terms.map((term, index) => {
    const strain = strainByKey[term.strain]
    const formula = strain?.formula ?? ""
    const label = `E_{${index + 1}}(\\lambda)`
    return subscriptStrainFormula(formula.replace(/^E\(\\lambda\)/, label), index + 1) || `${label} = E(\\lambda)`
  })
  return String.raw`\begin{aligned}${formulas.join(String.raw`\\`)}\end{aligned}`
}

function subscriptStrainFormula(formula, termIndex) {
  return formula
    .replace(/(^|[^A-Za-z\\])m(?![A-Za-z])/g, `$1m_{${termIndex}}`)
    .replace(/(^|[^A-Za-z\\])n(?![A-Za-z])/g, `$1n_{${termIndex}}`)
}

function hillParameters(model, terms) {
  const strainByKey = Object.fromEntries((model.configurable?.strains ?? []).map((strain) => [strain.key, strain]))
  return terms.flatMap((term, index) => {
    const termIndex = index + 1
    const strain = strainByKey[term.strain]
    return [
      { name: `mu${termIndex}`, initial: 10, bounds: [1e-6, null] },
      ...(strain?.parameters ?? []).map((param) => ({
        ...param,
        name: `${param.name}${termIndex}`,
      })),
    ]
  })
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
  const [solverSettings, setSolverSettings] = useState(defaultSolverSettings)
  const [optimization, setOptimization] = useState(initialOptimizationState)
  const [predictionSettings, setPredictionSettings] = useState(defaultPredictionSettings)
  const [predictionOverrides, setPredictionOverrides] = useState({})
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
        const modelByKey = Object.fromEntries(models.map((model) => [model.key, model]))
        const fallbackModel = models.find((item) => item.key === "ZhanNonGaussian") ?? models[0]
        setBranches((current) => current.map((branch) => ({
          ...branch,
          modelKey: modelByKey[branch.modelKey]?.key ?? (branch.modelKey?.startsWith("Ogden_") ? "Ogden" : null) ?? (branch.modelKey?.startsWith("Hill_") ? "Hill" : null) ?? fallbackModel?.key ?? "",
          modelConfig: normalizeConfigForModel(
            modelByKey[branch.modelKey] ?? modelByKey[branch.modelKey?.startsWith("Ogden_") ? "Ogden" : branch.modelKey?.startsWith("Hill_") ? "Hill" : ""] ?? fallbackModel,
            branch.modelConfig,
          ),
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
    () => buildConfiguredModel(modelCatalog.find((item) => item.key === selectedBranch?.modelKey) ?? modelCatalog[0], selectedBranch?.modelConfig),
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
  const optimizationParameters = useMemo(() => {
    return branches
      .filter((branch) => branch.enabled)
      .flatMap((branch) => {
        const model = buildConfiguredModel(modelCatalog.find((item) => item.key === branch.modelKey), branch.modelConfig)
        const overrides = parameterOverrides[branch.id] ?? {}
        return (model?.parameters ?? []).map((param) => ({
          key: `${branch.id}-${param.name}`,
          branch: branch.name,
          symbol: param.name,
          value: overrides[param.name] ?? String(param.initial ?? ""),
          lower: param.bounds?.[0],
          upper: param.bounds?.[1],
        }))
      })
  }, [branches, modelCatalog, parameterOverrides])

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
      modelConfig: defaultConfigForModel(modelCatalog.find((model) => model.key === modelKey)),
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

  function updateSolverSetting(name, value) {
    setSolverSettings((current) => ({ ...current, [name]: value }))
  }

  function startOptimization() {
    if (optimization.running) return
    setOptimization({
      status: "Running",
      running: true,
      progress: 4,
      initialLoss: 0.2468,
      currentLoss: 0.2468,
      r2: 0.0,
      iterations: 0,
      log: [
        `Started ${solverSettings.method} calibration.`,
        `${modes.length} dataset set${modes.length === 1 ? "" : "s"} and ${branches.filter((branch) => branch.enabled).length} active branch${branches.filter((branch) => branch.enabled).length === 1 ? "" : "es"} queued.`,
      ],
    })
  }

  function stopOptimization() {
    setOptimization((current) => ({
      ...current,
      status: "Stopped",
      running: false,
      log: [`Stopped at iteration ${current.iterations}.`, ...current.log].slice(0, 6),
    }))
  }

  function resetOptimization() {
    setOptimization(initialOptimizationState)
  }

  function updatePredictionSetting(name, value) {
    setPredictionSettings((current) => ({
      ...current,
      [name]: value,
      manualEdits: name === "source" && value === "manualOverride" ? true : current.manualEdits,
    }))
  }

  function updatePredictionOverride(key, value) {
    setPredictionOverrides((current) => {
      const next = { ...current }
      if (value === "") delete next[key]
      else next[key] = value
      return next
    })
  }

  function resetPredictionOverrides() {
    setPredictionOverrides({})
  }

  useEffect(() => {
    if (!optimization.running) return undefined
    const timer = window.setInterval(() => {
      setOptimization((current) => {
        if (!current.running) return current
        const iterations = current.iterations + 24
        const progress = Math.min(100, current.progress + 9)
        const currentLoss = Number(Math.max(0.0028, current.initialLoss * Math.exp(-progress / 27)).toFixed(6))
        const r2 = Number(Math.min(0.9987, 1 - currentLoss / current.initialLoss).toFixed(4))
        const finished = progress >= 100
        return {
          ...current,
          status: finished ? "Converged" : "Running",
          running: !finished,
          progress,
          iterations,
          currentLoss,
          r2,
          log: [
            finished
              ? `Converged after ${iterations} iterations.`
              : `Iteration ${iterations}: loss ${currentLoss.toExponential(2)}, R2 ${r2.toFixed(4)}.`,
            ...current.log,
          ].slice(0, 6),
        }
      })
    }, 650)
    return () => window.clearInterval(timer)
  }, [optimization.running])

  return (
    <div className="min-h-screen bg-background text-text-primary">
      <div className="grid min-h-screen grid-cols-[260px_minmax(0,1fr)]">
        <Sidebar activeStep={activeStep} onStepChange={setActiveStep} />
        <div className="flex min-w-0 flex-col">
          <Topbar activeStep={activeStep} />
          <main className="min-w-0 flex-1 overflow-x-hidden p-4 pb-24">
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
            ) : activeStep === "models" ? (
              <ModelArchitecturePage
                models={modelCatalog}
                branches={branches}
                selectedBranch={selectedBranch}
                selectedModel={selectedModel}
                parameterValues={selectedParameterValues}
                onAddBranch={addBranch}
                onRemoveBranch={removeBranch}
                onSelectBranch={setSelectedBranchId}
                onUpdateBranch={updateBranch}
                onParameterChange={handleParameterChange}
                onResetParameters={resetSelectedParameters}
                selectedDataCount={modes.length}
              />
            ) : activeStep === "optimization" ? (
              <OptimizationPage
                preview={preview}
                branches={branches}
                parameterRows={optimizationParameters}
                solverSettings={solverSettings}
                optimization={optimization}
                onSolverChange={updateSolverSetting}
                onStart={startOptimization}
                onStop={stopOptimization}
                onReset={resetOptimization}
              />
            ) : (
              <PredictionPage
                preview={preview}
                parameterRows={optimizationParameters}
                optimization={optimization}
                settings={predictionSettings}
                overrides={predictionOverrides}
                onSettingChange={updatePredictionSetting}
                onOverrideChange={updatePredictionOverride}
                onResetOverrides={resetPredictionOverrides}
              />
            )}
          </main>
          <BottomBar
            activeStep={activeStep}
            rows={preview.metadata?.rows ?? 0}
            selectedBranch={selectedBranch}
            selectedModel={selectedModel}
            onNext={() => {
              if (activeStep === "experimental") setActiveStep("models")
              else if (activeStep === "models") setActiveStep("optimization")
              else if (activeStep === "optimization") setActiveStep("prediction")
              else setActiveStep("optimization")
            }}
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
  onUpdateBranch,
  onParameterChange,
  onResetParameters,
  selectedDataCount,
}) {
  const modelByKey = useMemo(
    () => Object.fromEntries(models.map((model) => [model.key, model])),
    [models],
  )
  const selectedBaseModel = selectedBranch ? modelByKey[selectedBranch.modelKey] : null
  const activeBranches = branches.filter((branch) => branch.enabled)

  return (
    <div className="mx-auto flex min-h-[640px] w-full max-w-[1280px] flex-col gap-4 overflow-x-hidden">
      <ArchitectureSummary branches={branches} activeBranches={activeBranches} selectedDataCount={selectedDataCount} selectedBranch={selectedBranch} />
      <section className="grid min-h-[560px] min-w-0 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
        <Card title="Parallel Branches">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase text-text-muted">{activeBranches.length} active · {branches.length} total</p>
            <button className="flex shrink-0 items-center gap-1 rounded-lg bg-primary px-3 py-1.5 text-sm font-semibold text-white shadow-panel hover:bg-primary-hover" onClick={onAddBranch}>
              <Icon className="text-base">add</Icon>
              Add
            </button>
          </div>
          <div className="flex flex-col gap-2 pr-1">
            {branches.map((branch, index) => (
              <BranchCard
                key={branch.id}
                branch={branch}
                index={index}
                total={branches.length}
                model={buildConfiguredModel(modelByKey[branch.modelKey], branch.modelConfig)}
                active={branch.id === selectedBranch?.id}
                onSelect={() => onSelectBranch(branch.id)}
                onToggle={() => onUpdateBranch(branch.id, { enabled: !branch.enabled })}
                onRemove={() => onRemoveBranch(branch.id)}
              />
            ))}
          </div>
        </Card>

        <Card title="Architecture Workspace" className="min-w-0">
          <div className="grid min-w-0 gap-x-5 gap-y-5 lg:grid-cols-[minmax(0,1.08fr)_minmax(340px,0.92fr)]">
            <section className="min-w-0">
              <h4 className="text-xs font-bold uppercase text-text-muted">Parallel Structure</h4>
              <div className="mt-2 rounded-lg border border-border bg-subtle p-3">
                <ArchitectureVisualizer branches={branches} selectedBranchId={selectedBranch?.id} modelByKey={modelByKey} onSelectBranch={onSelectBranch} />
              </div>
            </section>

            <section className="min-w-0">
              <h4 className="text-xs font-bold uppercase text-text-muted">Selected Branch</h4>
              {selectedBranch ? (
                <div className="mt-2 space-y-3">
                  <label className="block">
                    <Label>Branch Name</Label>
                    <input className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={selectedBranch.name} onChange={(event) => onUpdateBranch(selectedBranch.id, { name: event.target.value })} />
                  </label>
                  <label className="block">
                    <Label>Constitutive Model</Label>
                    <select
                      className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary"
                      value={selectedBranch.modelKey}
                      onChange={(event) => {
                        const nextModel = modelByKey[event.target.value]
                        onUpdateBranch(selectedBranch.id, {
                          modelKey: event.target.value,
                          modelConfig: defaultConfigForModel(nextModel),
                        })
                      }}
                    >
                      {models.map((model) => (
                        <option key={model.key} value={model.key}>{model.name}</option>
                      ))}
                    </select>
                  </label>
                  <ModelConfigurationPanel
                    model={selectedBaseModel}
                    config={selectedBranch.modelConfig}
                    onChange={(modelConfig) => onUpdateBranch(selectedBranch.id, { modelConfig })}
                  />
                  <button className={`flex h-9 w-full items-center justify-center gap-1 rounded-lg border px-3 text-sm font-semibold ${selectedBranch.enabled ? "border-primary bg-selection-bg text-primary" : "border-border-strong bg-surface text-text-muted"}`} onClick={() => onUpdateBranch(selectedBranch.id, { enabled: !selectedBranch.enabled })}>
                    <Icon className="text-base">power_settings_new</Icon>
                    {selectedBranch.enabled ? "Enabled in Architecture" : "Disabled"}
                  </button>
                  {selectedModel && (
                    <div className="rounded-lg bg-subtle p-3">
                      <h3 className="text-sm font-semibold">{selectedModel.name}</h3>
                      <p className="mt-1 text-xs text-text-muted">{typeLabel(selectedModel.type)} · {selectedModel.category}{selectedModel.detail ? ` · ${selectedModel.detail}` : ""}</p>
                      {selectedModel.reference && <p className="mt-2 text-xs text-text-muted">Reference: {selectedModel.reference}</p>}
                    </div>
                  )}
                </div>
              ) : (
                <p className="mt-2 text-sm text-text-muted">No branch selected.</p>
              )}
            </section>

            <section className="min-w-0 border-t border-border pt-4">
              <h4 className="mb-2 text-xs font-bold uppercase text-text-muted">Model Equations</h4>
              {selectedModel ? (
                <div className="space-y-3">
                  <FormulaBlock label="Branch Energy Density" value={selectedModel.formula} />
                  {selectedModel.strainFormula && <FormulaBlock label="Generalized Strain" value={selectedModel.strainFormula} />}
                </div>
              ) : (
                <p className="text-sm text-text-muted">No model selected.</p>
              )}
            </section>

            <section className="min-w-0 border-t border-border pt-4">
              <h4 className="mb-2 text-xs font-bold uppercase text-text-muted">Branch Parameters</h4>
              <ParameterTable parameters={selectedModel?.parameters ?? []} values={parameterValues} onChange={onParameterChange} onReset={onResetParameters} />
            </section>
          </div>
        </Card>
      </section>
    </div>
  )
}

function OptimizationPage({
  preview,
  branches,
  parameterRows,
  solverSettings,
  optimization,
  onSolverChange,
  onStart,
  onStop,
  onReset,
}) {
  const activeBranches = branches.filter((branch) => branch.enabled)
  const metrics = [
    ["Initial loss", optimization.initialLoss.toExponential(2)],
    ["Current loss", optimization.currentLoss.toExponential(2)],
    ["R squared", optimization.r2.toFixed(4)],
    ["Iterations", optimization.iterations],
    ["Status", optimization.status],
  ]

  return (
    <div className="mx-auto flex min-h-[640px] w-full max-w-[1280px] flex-col gap-4 overflow-x-hidden">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        {metrics.map(([label, value]) => (
          <MetricCard key={label} label={label} value={value} active={label === "Status" && optimization.running} />
        ))}
      </div>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Card title="Solver Settings">
            <label className="block">
              <Label>Method</Label>
              <select className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={solverSettings.method} onChange={(event) => onSolverChange("method", event.target.value)}>
                {solverMethods.map((method) => <option key={method} value={method}>{method}</option>)}
              </select>
            </label>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <SolverNumber label="Max Iter." value={solverSettings.maxIter} onChange={(value) => onSolverChange("maxIter", value)} />
              <SolverNumber label="R2 Target" value={solverSettings.r2Target} step="0.001" onChange={(value) => onSolverChange("r2Target", value)} />
              <SolverNumber label="Abs. Tol." value={solverSettings.absTol} step="0.000001" onChange={(value) => onSolverChange("absTol", value)} />
              <SolverNumber label="Rel. Tol." value={solverSettings.relTol} step="0.0001" onChange={(value) => onSolverChange("relTol", value)} />
              <SolverNumber label="Max Loss" value={solverSettings.maxLoss} step="0.001" onChange={(value) => onSolverChange("maxLoss", value)} />
              <div className="rounded-lg border border-border bg-subtle p-2">
                <Label>Active Branches</Label>
                <div className="text-sm font-semibold">{activeBranches.length}</div>
              </div>
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-subtle">
              <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${optimization.progress}%` }} />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2">
              <button className="flex items-center justify-center gap-1 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white shadow-panel hover:bg-primary-hover disabled:opacity-50" onClick={onStart} disabled={optimization.running}>
                <Icon className="text-base">play_arrow</Icon>
                Start
              </button>
              <button className="flex items-center justify-center gap-1 rounded-lg border border-error px-3 py-2 text-sm font-semibold text-error hover:bg-red-50 disabled:opacity-40" onClick={onStop} disabled={!optimization.running}>
                <Icon className="text-base">stop</Icon>
                Stop
              </button>
              <button className="flex items-center justify-center gap-1 rounded-lg border border-border-strong px-3 py-2 text-sm font-semibold hover:bg-subtle" onClick={onReset}>
                <Icon className="text-base">restart_alt</Icon>
                Reset
              </button>
            </div>
          </Card>

          <Card title="Parameter Summary">
            <div className="overflow-hidden rounded-lg border border-border">
              <div className="grid grid-cols-[1fr_0.7fr_0.8fr] bg-subtle px-2 py-2 text-[11px] font-bold uppercase text-text-muted">
                <span>Parameter</span>
                <span>Initial</span>
                <span>Branch</span>
              </div>
              <div className="max-h-44 overflow-y-auto">
                {parameterRows.slice(0, 10).map((row) => (
                  <div key={row.key} className="grid grid-cols-[1fr_0.7fr_0.8fr] border-t border-border px-2 py-2 text-sm">
                    <span className="font-semibold"><LatexInline value={parameterSymbol(row.symbol)} fallback={row.symbol} /></span>
                    <span>{row.value || "-"}</span>
                    <span className="truncate text-text-muted">{row.branch}</span>
                  </div>
                ))}
              </div>
            </div>
            {parameterRows.length > 10 && <p className="mt-2 text-xs text-text-muted">+{parameterRows.length - 10} more parameters included in the solve.</p>}
          </Card>

          <Card title="Optimization Log">
            <div className="space-y-2">
              {optimization.log.map((entry, index) => (
                <div key={`${entry}-${index}`} className="rounded-md bg-subtle px-2 py-1.5 text-xs leading-5 text-text-secondary">
                  {entry}
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Card title="Calibration Fit">
            <div className="min-h-[460px]">
              <CalibrationFitChart preview={preview} progress={optimization.progress} />
            </div>
          </Card>

          <Card title="Fitted Parameters">
            <div className="grid grid-cols-[1fr_0.8fr_0.8fr_0.8fr] rounded-lg border border-border text-sm">
              <div className="contents text-[11px] font-bold uppercase text-text-muted">
                <div className="bg-subtle px-2 py-2">Name</div>
                <div className="bg-subtle px-2 py-2">Initial</div>
                <div className="bg-subtle px-2 py-2">Fitted</div>
                <div className="bg-subtle px-2 py-2">Bounds</div>
              </div>
              {parameterRows.slice(0, 6).map((row, index) => {
                const fitted = Number(row.value || 0) * (1 + 0.04 * Math.sin(index + optimization.progress / 40))
                return (
                  <div key={`${row.key}-fit`} className="contents">
                    <div className="border-t border-border px-2 py-2 font-semibold"><LatexInline value={parameterSymbol(row.symbol)} fallback={row.symbol} /></div>
                    <div className="border-t border-border px-2 py-2">{row.value || "-"}</div>
                    <div className="border-t border-border px-2 py-2">{Number.isFinite(fitted) ? fitted.toPrecision(4) : "-"}</div>
                    <div className="border-t border-border px-2 py-2 text-text-muted">{formatBound(row.lower)} / {formatBound(row.upper)}</div>
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      </section>
    </div>
  )
}

function MetricCard({ label, value, active }) {
  return (
    <div className={`rounded-lg border bg-surface p-3 shadow-panel ${active ? "border-primary" : "border-border"}`}>
      <div className="text-[11px] font-bold uppercase text-text-muted">{label}</div>
      <div className="mt-1 flex items-center gap-2 text-lg font-semibold">
        {active && <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />}
        {value}
      </div>
    </div>
  )
}

function SolverNumber({ label, value, step = "1", onChange }) {
  return (
    <label className="block">
      <Label>{label}</Label>
      <input className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" type="number" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function CalibrationFitChart({ preview, progress }) {
  const series = preview.series?.length
    ? preview.series
    : [{ modeFamily: "UT", modeLabel: "Experimental", points: preview.points ?? [] }]
  const allPoints = series.flatMap((item) => item.points ?? [])
  const xs = allPoints.map((point) => point.x)
  const ys = allPoints.map((point) => point.y)
  const minX = Math.min(...xs, 0)
  const maxX = Math.max(...xs, 1)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys, 1)
  const padX = Math.max((maxX - minX) * 0.08, 0.1)
  const padY = Math.max((maxY - minY) * 0.12, 0.1)
  const x0 = minX - padX
  const x1 = maxX + padX
  const y0 = minY - padY
  const y1 = maxY + padY
  const xScale = (x) => 8 + ((x - x0) / Math.max(x1 - x0, 1e-6)) * 88
  const yScale = (y) => 92 - ((y - y0) / Math.max(y1 - y0, 1e-6)) * 84
  const fitFactor = 0.82 + Math.min(progress, 100) * 0.0018

  return (
    <div className="relative h-full min-h-[460px] rounded-lg border border-border bg-white">
      <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" viewBox="0 0 100 100" role="img" aria-label="Calibration fit chart">
        {[20, 40, 60, 80].map((tick) => (
          <g key={tick}>
            <line x1={tick} x2={tick} y1="8" y2="92" stroke="#E5E5EA" strokeDasharray="1.5 1.5" strokeWidth="0.25" />
            <line x1="8" x2="96" y1={tick} y2={tick} stroke="#E5E5EA" strokeDasharray="1.5 1.5" strokeWidth="0.25" />
          </g>
        ))}
        <line x1="8" x2="8" y1="8" y2="92" stroke="#D1D1D6" strokeWidth="0.5" />
        <line x1="8" x2="96" y1="92" y2="92" stroke="#D1D1D6" strokeWidth="0.5" />
        {series.map((item, index) => {
          const points = item.points ?? []
          const fitPath = points
            .map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${xScale(point.x)} ${yScale(point.y * fitFactor)}`)
            .join(" ")
          return (
            <g key={item.mode ?? item.modeLabel ?? index}>
              <path d={fitPath} fill="none" stroke={colorForSeries(item.modeFamily, index)} strokeWidth="0.8" />
              {points.map((point, pointIndex) => (
                <circle key={pointIndex} cx={xScale(point.x)} cy={yScale(point.y)} r="0.8" fill="#FFFFFF" stroke={colorForSeries(item.modeFamily, index)} strokeWidth="0.45" />
              ))}
            </g>
          )
        })}
      </svg>
      <div className="absolute left-3 top-3 rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
        <div className="font-semibold">Experimental vs model fit</div>
        <div className="mt-1 text-text-muted">{preview.metadata?.selectedMode ?? "Selected datasets"}</div>
      </div>
      <div className="absolute bottom-3 right-3 rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
        <div className="flex items-center gap-2"><span className="h-2 w-4 rounded-full bg-primary" /> Model fit</div>
        <div className="mt-1 flex items-center gap-2"><span className="h-2 w-2 rounded-full border border-primary bg-white" /> Experimental</div>
      </div>
    </div>
  )
}

function PredictionPage({
  preview,
  parameterRows,
  optimization,
  settings,
  overrides,
  onSettingChange,
  onOverrideChange,
  onResetOverrides,
}) {
  const changedCount = Object.keys(overrides).length
  const prediction = useMemo(
    () => buildPredictionSeries(preview, parameterRows, settings, overrides, optimization),
    [preview, parameterRows, settings, overrides, optimization],
  )
  const peakStress = prediction.curve.reduce((max, point) => Math.max(max, point.y), 0)
  const maxExperimentalX = prediction.experimental.reduce((max, point) => Math.max(max, point.x), 0)
  const extrapolation = Math.max(0, Number(settings.stretchMax) - maxExperimentalX)
  const metrics = [
    ["Peak stress", `${peakStress.toFixed(3)} MPa`],
    ["Stretch range", `${Number(settings.stretchMin).toFixed(2)}-${Number(settings.stretchMax).toFixed(2)}`],
    ["Samples", settings.samples],
    ["Extrapolation", `${extrapolation.toFixed(2)} λ`],
    ["Status", settings.manualEdits ? "Manual edits" : "Ready"],
  ]

  return (
    <div className="mx-auto flex min-h-[640px] w-full max-w-[1280px] flex-col gap-4 overflow-x-hidden">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        {metrics.map(([label, value]) => (
          <MetricCard key={label} label={label} value={value} active={label === "Status" && settings.manualEdits} />
        ))}
      </div>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Card title="Parameter Source">
            <div className="space-y-2">
              {[
                ["lastOptimization", "Last Optimization", `R2 ${optimization.r2.toFixed(4)} · ${optimization.status}`],
                ["currentArchitecture", "Current Architecture", `${parameterRows.length} initial parameters`],
                ["manualOverride", "Manual Override", `${changedCount} changed value${changedCount === 1 ? "" : "s"}`],
              ].map(([value, label, detail]) => (
                <button
                  key={value}
                  className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left ${settings.source === value ? "border-primary bg-selection-bg" : "border-border-strong hover:bg-subtle"}`}
                  onClick={() => onSettingChange("source", value)}
                >
                  <span className={`h-3 w-3 rounded-full ${settings.source === value ? "bg-primary" : "bg-border-strong"}`} />
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold">{label}</span>
                    <span className="block truncate text-xs text-text-muted">{detail}</span>
                  </span>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Prediction Setup">
            <label className="block">
              <Label>Loading Mode</Label>
              <select className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={settings.mode} onChange={(event) => onSettingChange("mode", event.target.value)}>
                {modeOptions.map((mode) => <option key={mode.family} value={mode.family}>{mode.label}</option>)}
              </select>
            </label>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <SolverNumber label="Stretch Min" value={settings.stretchMin} step="0.05" onChange={(value) => onSettingChange("stretchMin", value)} />
              <SolverNumber label="Stretch Max" value={settings.stretchMax} step="0.05" onChange={(value) => onSettingChange("stretchMax", value)} />
              <SolverNumber label="Samples" value={settings.samples} step="1" onChange={(value) => onSettingChange("samples", value)} />
              <label className="block">
                <Label>Stress Output</Label>
                <select className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={settings.stressOutput} onChange={(event) => onSettingChange("stressOutput", event.target.value)}>
                  <option>First PK</option>
                  <option>Cauchy</option>
                  <option>Engineering</option>
                </select>
              </label>
            </div>
          </Card>

          <Card title="Manual Overrides">
            <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-subtle px-3 py-2">
              <span>
                <span className="block text-sm font-semibold">Enable manual edits</span>
                <span className="block text-xs text-text-muted">Overrides are isolated from fitted values.</span>
              </span>
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-border-strong text-primary focus:ring-primary"
                checked={settings.manualEdits}
                onChange={(event) => onSettingChange("manualEdits", event.target.checked)}
              />
            </label>
            <div className="mt-3 overflow-hidden rounded-lg border border-border">
              <div className="grid grid-cols-[1fr_0.8fr_0.9fr] bg-subtle px-2 py-2 text-[11px] font-bold uppercase text-text-muted">
                <span>Parameter</span>
                <span>Fitted</span>
                <span>Override</span>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {parameterRows.slice(0, 12).map((row) => {
                  const overrideValue = overrides[row.key] ?? ""
                  return (
                    <div key={row.key} className="grid grid-cols-[1fr_0.8fr_0.9fr] items-center gap-2 border-t border-border px-2 py-2 text-sm">
                      <span className="min-w-0 truncate font-semibold">
                        <LatexInline value={parameterSymbol(row.symbol)} fallback={row.symbol} />
                      </span>
                      <span className="truncate text-text-muted">{row.value || "-"}</span>
                      <input
                        className="h-8 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary disabled:bg-subtle disabled:text-text-disabled"
                        type="number"
                        step="any"
                        disabled={!settings.manualEdits}
                        placeholder={String(row.value ?? "")}
                        value={overrideValue}
                        onChange={(event) => onOverrideChange(row.key, event.target.value)}
                      />
                    </div>
                  )
                })}
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between gap-3">
              <div className="flex flex-wrap gap-1">
                {changedCount ? (
                  Object.keys(overrides).slice(0, 4).map((key) => {
                    const row = parameterRows.find((item) => item.key === key)
                    return <span key={key} className="rounded border border-primary bg-selection-bg px-1.5 py-0.5 text-[11px] font-semibold text-primary">{row?.symbol ?? "param"}</span>
                  })
                ) : (
                  <span className="text-xs text-text-muted">No overrides applied.</span>
                )}
              </div>
              <button className="rounded-lg border border-border-strong px-3 py-1.5 text-sm font-semibold hover:bg-subtle" onClick={onResetOverrides}>
                Reset
              </button>
            </div>
          </Card>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Card title="Prediction Plot">
            <div className="min-h-[540px]">
              <PredictionChart prediction={prediction} settings={settings} preview={preview} />
            </div>
          </Card>

          <Card title="Parameter Provenance">
            <div className="grid gap-2 sm:grid-cols-3">
              {[
                ["Source", sourceLabel(settings.source)],
                ["Manual changed", changedCount],
                ["Optimization status", `${optimization.status} · ${optimization.iterations} iter.`],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-border bg-subtle p-2">
                  <div className="text-[11px] font-bold uppercase text-text-muted">{label}</div>
                  <div className="mt-1 truncate text-sm font-semibold">{value}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </section>
    </div>
  )
}

function PredictionChart({ prediction, settings, preview }) {
  const points = [...prediction.experimental, ...prediction.curve]
  const width = 760
  const height = 500
  const pad = { top: 34, right: 28, bottom: 60, left: 78 }
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs, 1)
  const maxX = Math.max(...xs, 3)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys, 1)
  const xScale = (x) => pad.left + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (width - pad.left - pad.right)
  const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (height - pad.top - pad.bottom)
  const curvePath = prediction.curve.map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x)} ${yScale(point.y)}`).join(" ")
  const experimentalMaxX = prediction.experimental.reduce((max, point) => Math.max(max, point.x), minX)
  const extrapolationX = xScale(experimentalMaxX)

  return (
    <div className="relative h-full min-h-[540px] rounded-lg border border-border bg-white">
      <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Prediction chart">
        <rect width={width} height={height} fill="#FFFFFF" />
        <rect x={extrapolationX} y={pad.top} width={width - pad.right - extrapolationX} height={height - pad.top - pad.bottom} fill="#EAF3FF" opacity="0.7" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const x = pad.left + tick * (width - pad.left - pad.right)
          const y = pad.top + tick * (height - pad.top - pad.bottom)
          return (
            <g key={tick}>
              <line x1={x} x2={x} y1={pad.top} y2={height - pad.bottom} stroke="#E5E5EA" strokeDasharray="4 4" />
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#E5E5EA" strokeDasharray="4 4" />
            </g>
          )
        })}
        <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} stroke="#D1D1D6" />
        <line x1={pad.left} x2={pad.left} y1={pad.top} y2={height - pad.bottom} stroke="#D1D1D6" />
        <path d={curvePath} fill="none" stroke="#007AFF" strokeWidth="3" />
        {prediction.experimental.map((point, index) => (
          <circle key={`${point.x}-${point.y}-${index}`} cx={xScale(point.x)} cy={yScale(point.y)} r="4" fill="#FFFFFF" stroke={colorForSeries(point.family, 0)} strokeWidth="2" />
        ))}
        <text x={width / 2} y={height - 18} textAnchor="middle" fontSize="12" fill="#6E6E73">
          Stretch lambda (-)
        </text>
        <text x="18" y={height / 2} textAnchor="middle" fontSize="12" fill="#6E6E73" transform={`rotate(-90 18 ${height / 2})`}>
          {settings.stressOutput} Stress (MPa)
        </text>
      </svg>
      <div className="absolute left-3 top-3 rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
        <div className="font-semibold">Prediction review</div>
        <div className="mt-1 text-text-muted">{preview.metadata?.selectedMode ?? "Selected datasets"}</div>
      </div>
      <div className="absolute bottom-3 right-3 rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
        <div className="flex items-center gap-2"><span className="h-2 w-5 rounded-full bg-primary" /> Prediction curve</div>
        <div className="mt-1 flex items-center gap-2"><span className="h-3 w-3 rounded-full border-2 border-primary bg-white" /> Calibration data</div>
        <div className="mt-1 flex items-center gap-2"><span className="h-3 w-5 rounded-sm bg-selection-bg" /> Extrapolation</div>
      </div>
    </div>
  )
}

function buildPredictionSeries(preview, parameterRows, settings, overrides, optimization) {
  const series = preview.series?.length
    ? preview.series
    : [{ modeFamily: "UT", modeLabel: "Experimental", points: preview.points ?? [] }]
  const experimental = series.flatMap((item) => (item.points ?? []).map((point) => ({ ...point, family: item.modeFamily })))
  const values = parameterRows.map((row, index) => {
    const override = overrides[row.key]
    const value = settings.manualEdits && override !== "" && override !== undefined ? override : row.value
    const numeric = Number(value)
    return Number.isFinite(numeric) ? Math.abs(numeric) / (index + 3) : 0
  })
  const parameterScale = values.reduce((sum, value) => sum + value, 0)
  const sourceBoost = settings.source === "lastOptimization" ? 1 + optimization.progress / 700 : settings.source === "manualOverride" ? 1.08 : 0.96
  const stiffness = Math.max(0.28, Math.min(3.6, (0.72 + parameterScale * 0.035) * sourceBoost))
  const min = Number(settings.stretchMin) || 1
  const max = Math.max(min + 0.05, Number(settings.stretchMax) || 3.5)
  const sampleCount = Math.max(12, Math.min(240, Number(settings.samples) || 72))
  const modeFactor = {
    CSS: 0.88,
    UT: 1,
    PS: 1.16,
    ET: 1.34,
    SS: 0.76,
    BT: 1.24,
  }[settings.mode] ?? 1
  const curve = Array.from({ length: sampleCount }, (_, index) => {
    const t = sampleCount === 1 ? 0 : index / (sampleCount - 1)
    const lambda = min + t * (max - min)
    const strain = Math.max(0, lambda - 1)
    const y = stiffness * modeFactor * strain * (1 + 0.52 * strain * strain)
    return { x: lambda, y }
  })
  return { experimental, curve }
}

function sourceLabel(source) {
  return {
    lastOptimization: "Last Optimization",
    currentArchitecture: "Current Architecture",
    manualOverride: "Manual Override",
  }[source] ?? source
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

function BranchCard({ branch, total, model, active, onSelect, onToggle, onRemove }) {
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
          <div className="mt-1 truncate text-xs text-text-muted">{model?.name ?? branch.modelKey}{model?.detail ? ` · ${model.detail}` : ""}</div>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-1">
        <IconButton label={branch.enabled ? "Disable" : "Enable"} onClick={onToggle}>power_settings_new</IconButton>
        <IconButton label="Remove" disabled={total === 1} onClick={onRemove}>delete</IconButton>
      </div>
    </div>
  )
}

function ModelConfigurationPanel({ model, config, onChange }) {
  const kind = model?.configurable?.kind
  if (!kind) return null

  const normalized = normalizeConfigForModel(model, config)
  if (kind === "ogden") {
    const min = model.configurable?.minTerms ?? 1
    const max = model.configurable?.maxTerms ?? 5
    return (
      <label className="block">
        <Label>Terms</Label>
        <select
          className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary"
          value={normalized.termCount}
          onChange={(event) => onChange({ termCount: Number(event.target.value) })}
        >
          {Array.from({ length: max - min + 1 }, (_, index) => min + index).map((count) => (
            <option key={count} value={count}>{count} term{count === 1 ? "" : "s"}</option>
          ))}
        </select>
      </label>
    )
  }

  const strains = model.configurable?.strains ?? []
  const min = model.configurable?.minTerms ?? 1
  const max = model.configurable?.maxTerms ?? 5
  const terms = normalized.terms
  const updateTerm = (termIndex, strain) => {
    onChange({ terms: terms.map((term, index) => index === termIndex ? { strain } : term) })
  }
  const addTerm = () => {
    if (terms.length >= max) return
    onChange({ terms: [...terms, { strain: strains[0]?.key ?? "" }] })
  }
  const removeTerm = (termIndex) => {
    if (terms.length <= min) return
    onChange({ terms: terms.filter((_, index) => index !== termIndex) })
  }

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <Label>Hill Terms</Label>
        <button className="flex items-center gap-1 rounded-md border border-border-strong bg-surface px-2 py-1 text-xs font-semibold hover:bg-subtle disabled:opacity-40" disabled={terms.length >= max} onClick={addTerm}>
          <Icon className="text-sm">add</Icon>
          Term
        </button>
      </div>
      <div className="space-y-2">
        {terms.map((term, index) => (
          <div key={`${term.strain}-${index}`} className="grid grid-cols-[auto_1fr_auto] items-center gap-2 rounded-lg border border-border bg-subtle p-2">
            <span className="text-xs font-bold text-text-muted">{index + 1}</span>
            <select className="h-8 min-w-0 rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary" value={term.strain} onChange={(event) => updateTerm(index, event.target.value)}>
              {strains.map((strain) => (
                <option key={strain.key} value={strain.key}>{strain.name}</option>
              ))}
            </select>
            <button className="grid h-8 w-8 place-items-center rounded-md border border-border-strong bg-surface text-text-muted hover:bg-subtle disabled:opacity-40" disabled={terms.length <= min} title="Remove term" onClick={() => removeTerm(index)}>
              <Icon className="text-sm">delete</Icon>
            </button>
          </div>
        ))}
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
    <div className="overflow-x-auto">
      <svg className="h-auto min-h-[260px] w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Parallel spring architecture">
        <rect x="0" y="0" width={width} height={height} rx="10" fill="#FFFFFF" />
        <line x1={railLeft} y1={top - 24} x2={railLeft} y2={top + (branches.length - 1) * rowGap + 24} stroke="#717786" strokeWidth="4" strokeLinecap="round" />
        <line x1={railRight} y1={top - 24} x2={railRight} y2={top + (branches.length - 1) * rowGap + 24} stroke="#717786" strokeWidth="4" strokeLinecap="round" />
        <text x={railLeft} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#6E6E73">F input</text>
        <text x={railRight} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#6E6E73">P total</text>
        {branches.map((branch, index) => {
          const model = buildConfiguredModel(modelByKey[branch.modelKey], branch.modelConfig)
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
              <text x={springStart + 126} y={y + 13} fontSize="11" fill="#6E6E73">{model?.name ?? branch.modelKey}{model?.detail ? ` · ${model.detail}` : ""}</text>
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
    ["optimization", "tune", "Optimization", "ready"],
    ["prediction", "analytics", "Prediction", "ready"],
  ]
  return (
    <aside className="sticky top-0 flex h-screen flex-col border-r border-border bg-surface px-3 py-4">
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
  const title = {
    experimental: "Experimental Data Workspace",
    models: "Model Architecture",
    optimization: "Optimization",
    prediction: "Prediction",
  }[activeStep] ?? "Experimental Data Workspace"
  const subtitle = {
    experimental: "Project: Experimental Data Workspace",
    models: "Constitutive model selection and parameter setup",
    optimization: "Calibration run and convergence review",
    prediction: "Fitted parameter reuse and prediction review",
  }[activeStep] ?? "Project: Experimental Data Workspace"
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
    <div className={`min-w-0 rounded-lg border border-border bg-surface p-3 shadow-panel ${className}`}>
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
  const dataStatus = {
    Ready: "Loaded",
    Connecting: "Connecting",
    Offline: "Offline",
    "Preview error": "Preview error",
  }[apiState] ?? apiState
  return (
    <dl className="grid grid-cols-2 gap-2 text-sm">
      <MetaItem label="Rows parsed" value={metadata.rows ?? 0} />
      <MetaItem label="Data status" value={dataStatus} />
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
    : activeStep === "optimization"
      ? "Optimization workspace ready"
      : activeStep === "prediction"
        ? "Prediction workspace ready"
        : `Status: Ready · ${rows} rows parsed`
  const nextLabel = {
    experimental: "Next Step",
    models: "Optimize",
    optimization: "Prediction",
    prediction: "Back to Optimization",
  }[activeStep] ?? "Next Step"
  return (
    <footer className="sticky bottom-0 z-20 flex h-16 shrink-0 items-center justify-between border-t border-border bg-surface/95 px-4 backdrop-blur">
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
