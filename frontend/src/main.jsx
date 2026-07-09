import React, { useEffect, useMemo, useRef, useState } from "react"
import { createRoot } from "react-dom/client"
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Copy,
  Database,
  FlaskConical,
  Home,
  Info,
  Layers,
  LineChart,
  Plus,
  Power,
  Play,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Trash2,
  ZoomIn,
} from "lucide-react"
import katex from "katex"
import "katex/dist/katex.min.css"
import "./styles.css"

// Same-origin by default: the FastAPI server also hosts the built frontend
// (desktop app / production). `npm run dev` proxies /api via vite.config.js.
const API_BASE = import.meta.env.VITE_API_BASE ?? ""
const defaultBranches = [
  { id: "spring-1", name: "Spring 1", modelKey: "ZhanNonGaussian", enabled: true },
]
const hiddenDataFamilies = new Set(["CSS"])
const fittingDisabledFamilies = new Set(["BT"])

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
  info: Info,
  monitoring: Activity,
  add: Plus,
  play_arrow: Play,
  power_settings_new: Power,
  restart_alt: RotateCcw,
  save: Save,
  schema: Layers,
  science: FlaskConical,
  stop: Square,
  delete: Trash2,
  tune: SlidersHorizontal,
  zoom_in: ZoomIn,
  copy: Copy,
  home: Home,
  search: Search,
  check: CheckCircle2,
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

const chartTheme = {
  width: 720,
  height: 400,
  pad: { top: 24, right: 24, bottom: 52, left: 78 },
  ticks: [0, 0.25, 0.5, 0.75, 1],
  grid: "#E5E5EA",
  axis: "#C7C7CC",
  tick: "#8E8E93",
  label: "#1C1C1E",
  gridDash: "4 4",
  tickFont: 11,
  lineWidth: 2.2,
  fitLineWidth: 2.4,
  pointRadius: 3.3,
  pointStrokeWidth: 1.2,
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
  modeKeys: [],
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

function enforceCalibrationSelection(keys, authorModes) {
  return keys.filter((key) => !fittingDisabledFamilies.has(authorModes.find((item) => item.key === key)?.family))
}

function visibleDataModes(modes = []) {
  return modes.filter((item) => !hiddenDataFamilies.has(item.family))
}

function visibleDatasetAuthors(authors = []) {
  return authors
    .map((item) => ({ ...item, modes: visibleDataModes(item.modes ?? []) }))
    .filter((item) => item.modes.length)
}

function preferredPredictionKeys(authorModes, fittedKeys = []) {
  const fittedSet = new Set(fittedKeys)
  const biaxial = authorModes.filter((item) => item.family === "BT" && !fittedSet.has(item.key))
  if (biaxial.length) return biaxial.map((item) => item.key)
  return authorModes.filter((item) => !fittedSet.has(item.key)).slice(0, 1).map((item) => item.key)
}

function defaultFittingKeys(authorRecord) {
  const modes = authorRecord?.modes ?? []
  const fittingModes = modes.filter((item) => !fittingDisabledFamilies.has(item.family))
  if (authorRecord?.author === "Treloar_1944") {
    return fittingModes.map((item) => item.key)
  }
  const fallback = fittingModes.find((item) => item.family === "UT") ?? fittingModes[0] ?? modes[0]
  return fallback ? [fallback.key] : []
}

function buddayRegionOptions(author, authorModes) {
  if (author !== "Budday_2017") return []
  const byKey = new Map()
  authorModes.forEach((mode) => {
    if (mode.tissueRegion && !byKey.has(mode.tissueRegion)) {
      byKey.set(mode.tissueRegion, {
        key: mode.tissueRegion,
        label: mode.tissueRegionLabel ?? formatDisplayLabel(mode.tissueRegion),
      })
    }
  })
  return Array.from(byKey.values()).sort((left, right) => left.label.localeCompare(right.label))
}

function fittingModeOptions(author, authorModes, buddayRegion) {
  if (author !== "Budday_2017" || !buddayRegion) return authorModes
  return authorModes.filter((mode) => mode.tissueRegion === buddayRegion)
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

function downsamplePoints(points, target) {
  const source = (points ?? []).filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y))
  if (source.length <= target) return source.map((point) => [point.x, point.y])
  const step = (source.length - 1) / (target - 1)
  return Array.from({ length: target }, (_, index) => {
    const point = source[Math.round(index * step)]
    return [point.x, point.y]
  })
}

function sessionDateGroup(updatedAt) {
  const now = new Date()
  const date = new Date(updatedAt * 1000)
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  if (date.getTime() >= startOfToday) return "Today"
  if (date.getTime() >= startOfToday - 6 * 24 * 3600 * 1000) return "Earlier this week"
  return "Earlier"
}

function sessionDateLabel(updatedAt) {
  const date = new Date(updatedAt * 1000)
  const now = new Date()
  const hm = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  if (date.getTime() >= startOfToday) return `Today ${hm}`
  if (date.getTime() >= startOfToday - 24 * 3600 * 1000) return `Yesterday ${hm}`
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

function App() {
  const [view, setView] = useState("start")
  const [sessions, setSessions] = useState(null) // null = loading
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [myDatasets, setMyDatasets] = useState([])
  const [activeStep, setActiveStep] = useState("experimental")
  const [datasets, setDatasets] = useState([])
  const [author, setAuthor] = useState("")
  const [modes, setModes] = useState([])
  const [buddayRegion, setBuddayRegion] = useState("")
  const [modelCatalog, setModelCatalog] = useState([])
  const [branches, setBranches] = useState(defaultBranches)
  const [selectedBranchId, setSelectedBranchId] = useState(defaultBranches[0].id)
  const [parameterOverrides, setParameterOverrides] = useState({})
  const [fittedParameterValues, setFittedParameterValues] = useState({})
  const [solverSettings, setSolverSettings] = useState(defaultSolverSettings)
  const [optimization, setOptimization] = useState(initialOptimizationState)
  const [predictionSettings, setPredictionSettings] = useState(defaultPredictionSettings)
  const [predictionOverrides, setPredictionOverrides] = useState({})
  const [preview, setPreview] = useState({
    points: samplePoints,
    axes: { x: "Stretch λ (-)", y: "Nominal stress P₁₁ (MPa)" },
    metadata: {
      rows: samplePoints.length,
      source: "Waiting for database",
      selectedMode: "Uniaxial Tension",
      stressType: "First PK",
      setCount: 1,
    },
  })
  const [predictionPreview, setPredictionPreview] = useState(preview)
  // Set once a session restore has been applied (or is pending), so the boot
  // effect does not clobber restored selections with defaults.
  const restoredRef = useRef(false)

  useEffect(() => {
    Promise.all([fetch(`${API_BASE}/api/datasets`), fetch(`${API_BASE}/api/models`)])
      .then(async ([datasetRes, modelRes]) => {
        const datasetData = await datasetRes.json()
        const modelData = await modelRes.json()
        const authors = visibleDatasetAuthors(datasetData.authors ?? [])
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
        if (!restoredRef.current) {
          const preferred = authors.find((item) => item.author === "Treloar_1944") ?? authors[0]
          if (preferred) {
            setAuthor(preferred.author)
            setModes(defaultFittingKeys(preferred))
          }
        }
      })
      .catch(() => {})
    refreshMyDatasets()
  }, [])

  async function refreshDatasets() {
    const response = await fetch(`${API_BASE}/api/datasets`)
    const data = await response.json()
    const authors = visibleDatasetAuthors(data.authors ?? [])
    setDatasets(authors)
    return authors
  }

  async function refreshModels() {
    try {
      const response = await fetch(`${API_BASE}/api/models`)
      const data = await response.json()
      setModelCatalog(data.models ?? [])
      return data.models ?? []
    } catch {
      return modelCatalog
    }
  }

  async function refreshMyDatasets() {
    try {
      const response = await fetch(`${API_BASE}/api/user-datasets`)
      const data = await response.json()
      setMyDatasets(data.datasets ?? [])
    } catch {
      setMyDatasets([])
    }
  }

  async function handleUploaded(newAuthor, modeKey) {
    refreshMyDatasets()
    try {
      await refreshDatasets()
    } catch {
      return
    }
    setBuddayRegion("")
    setAuthor(newAuthor)
    setModes([modeKey])
  }

  async function handleDeleteUserDataset(datasetId) {
    try {
      await fetch(`${API_BASE}/api/user-datasets/${datasetId}`, { method: "DELETE" })
    } catch {
      /* ignore */
    }
    refreshMyDatasets()
    try {
      const authors = await refreshDatasets()
      if (!authors.some((item) => item.author === author)) {
        const preferred = authors.find((item) => item.author === "Treloar_1944") ?? authors[0]
        if (preferred) {
          setBuddayRegion("")
          setAuthor(preferred.author)
          setModes(defaultFittingKeys(preferred))
        }
      }
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    if (view !== "start") return
    fetch(`${API_BASE}/api/sessions`)
      .then((res) => res.json())
      .then((data) => setSessions(data.sessions ?? []))
      .catch(() => setSessions([]))
  }, [view])

  useEffect(() => {
    if (!author || !modes.length) return
    const params = new URLSearchParams({ author })
    modes.forEach((item) => params.append("mode", item))
    fetch(`${API_BASE}/api/preview?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        setPreview(data)
      })
      .catch(() => {})
  }, [author, modes])

  const authorRecord = useMemo(() => datasets.find((item) => item.author === author), [author, datasets])
  const authorModes = authorRecord?.modes ?? []
  const buddayRegions = useMemo(() => buddayRegionOptions(author, authorModes), [author, authorModes])
  const fittingModes = useMemo(() => fittingModeOptions(author, authorModes, buddayRegion), [author, authorModes, buddayRegion])
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
        return (model?.parameters ?? []).map((param) => {
          const key = `${branch.id}-${param.name}`
          const initial = overrides[param.name] ?? String(param.initial ?? "")
          return {
            key,
            branch: branch.name,
            symbol: param.name,
            initial,
            value: fittedParameterValues[key] ?? initial,
            fitted: fittedParameterValues[key],
            lower: param.bounds?.[0],
            upper: param.bounds?.[1],
          }
        })
      })
  }, [branches, fittedParameterValues, modelCatalog, parameterOverrides])

  useEffect(() => {
    if (!authorModes.length) return
    const validKeys = new Set(authorModes.map((item) => item.key))
    const validSelection = modes.filter((key) => validKeys.has(key))
    if (!validSelection.length) {
      const fallback = authorModes.find((item) => item.family === "UT") ?? authorModes.find((item) => !fittingDisabledFamilies.has(item.family)) ?? authorModes[0]
      setModes([fallback.key])
      return
    }
    const consistentSelection = enforceCalibrationSelection(validSelection, authorModes)
    if (!consistentSelection.length) {
      const fallback = authorModes.find((item) => item.family === "UT") ?? authorModes.find((item) => !fittingDisabledFamilies.has(item.family)) ?? authorModes[0]
      setModes(fallback ? [fallback.key] : [])
      return
    }
    if (consistentSelection.join("|") !== modes.join("|")) {
      setModes(consistentSelection)
    }
  }, [author, authorModes, modes])

  useEffect(() => {
    if (author !== "Budday_2017") return
    if (!buddayRegions.length) return
    const nextRegion = buddayRegions.some((region) => region.key === buddayRegion)
      ? buddayRegion
      : buddayRegions[0].key
    if (nextRegion !== buddayRegion) {
      setBuddayRegion(nextRegion)
      return
    }
    const regionModes = fittingModeOptions(author, authorModes, nextRegion).filter((mode) => !fittingDisabledFamilies.has(mode.family))
    if (regionModes.length && !modes.every((key) => regionModes.some((mode) => mode.key === key))) {
      const fallback = regionModes.find((mode) => mode.family === "UT") ?? regionModes[0]
      setModes([fallback.key])
    }
  }, [author, authorModes, buddayRegion, buddayRegions, modes])

  useEffect(() => {
    if (!authorModes.length) return
    const validKeys = new Set(authorModes.map((item) => item.key))
    const currentKeys = (predictionSettings.modeKeys ?? []).filter((key) => validKeys.has(key))
    const fitted = modes.filter((key) => validKeys.has(key))
    const fallbackKeys = preferredPredictionKeys(authorModes, fitted)
    const nextKeys = currentKeys.length
      ? currentKeys
      : fallbackKeys.length
        ? fallbackKeys
        : [authorModes.find((item) => item.family === "UT")?.key ?? authorModes[0].key]
    if (nextKeys.join("|") !== (predictionSettings.modeKeys ?? []).join("|")) {
      setPredictionSettings((current) => ({ ...current, modeKeys: nextKeys }))
    }
  }, [authorModes, modes, predictionSettings.modeKeys])

  useEffect(() => {
    const modeKeys = predictionSettings.modeKeys ?? []
    if (!author || !modeKeys.length) return
    const params = new URLSearchParams({ author })
    modeKeys.forEach((item) => params.append("mode", item))
    fetch(`${API_BASE}/api/preview?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => setPredictionPreview(data))
      .catch(() => setPredictionPreview(preview))
  }, [author, predictionSettings.modeKeys, preview])

  function handleAuthorChange(value) {
    setAuthor(value)
    const record = datasets.find((item) => item.author === value)
    const regions = buddayRegionOptions(value, record?.modes ?? [])
    const nextRegion = regions[0]?.key ?? ""
    setBuddayRegion(nextRegion)
    const options = fittingModeOptions(value, record?.modes ?? [], nextRegion)
    setModes(defaultFittingKeys({ ...record, modes: options }))
  }

  function handleBuddayRegionChange(value) {
    setBuddayRegion(value)
    const options = fittingModeOptions(author, authorModes, value).filter((mode) => !fittingDisabledFamilies.has(mode.family))
    const fallback = options.find((item) => item.family === "UT") ?? options[0]
    setModes(fallback ? [fallback.key] : [])
  }

  function handleModeClick(modeRecord) {
    if (modeRecord.family === "BT") {
      window.alert("Biaxial tension data are reserved for prediction and cannot be used as fitting data.")
      return
    }
    setModes((current) => {
      if (current.includes(modeRecord.key)) {
        return current.length === 1 ? current : current.filter((key) => key !== modeRecord.key)
      }
      const currentFamily = authorModes.find((item) => item.key === current[0])?.family
      if ((currentFamily === "BT" && modeRecord.family !== "BT") || (currentFamily !== "BT" && modeRecord.family === "BT")) {
        return [modeRecord.key]
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

  function buildCalibrationPayload() {
    const activeBranches = branches.filter((branch) => branch.enabled).map((branch) => {
      const model = buildConfiguredModel(modelCatalog.find((item) => item.key === branch.modelKey), branch.modelConfig)
      const overrides = parameterOverrides[branch.id] ?? {}
      const parameters = Object.fromEntries((model?.parameters ?? []).map((param) => [
        param.name,
        Number(overrides[param.name] ?? fittedParameterValues[`${branch.id}-${param.name}`] ?? param.initial ?? 0),
      ]))
      return {
        id: branch.id,
        name: branch.name,
        modelKey: branch.modelKey,
        modelConfig: model?.config ?? branch.modelConfig ?? {},
        enabled: branch.enabled,
        parameters,
      }
    })
    return {
      author,
      modes,
      branches: activeBranches,
      solver: solverSettings,
    }
  }

  function buildSessionSnapshot(overrides = {}) {
    return {
      activeStep,
      author,
      buddayRegion,
      modes,
      branches,
      selectedBranchId,
      parameterOverrides,
      fittedParameterValues,
      solverSettings,
      optimization: { ...optimization, running: false },
      predictionSettings,
      predictionOverrides,
      ...overrides,
    }
  }

  function applySessionState(state) {
    restoredRef.current = true
    setAuthor(state.author ?? "")
    setBuddayRegion(state.buddayRegion ?? "")
    setModes(Array.isArray(state.modes) ? state.modes : [])
    const restoredBranches = Array.isArray(state.branches) && state.branches.length ? state.branches : defaultBranches
    setBranches(restoredBranches)
    setSelectedBranchId(state.selectedBranchId ?? restoredBranches[0].id)
    setParameterOverrides(state.parameterOverrides ?? {})
    setFittedParameterValues(state.fittedParameterValues ?? {})
    setSolverSettings({ ...defaultSolverSettings, ...(state.solverSettings ?? {}) })
    setOptimization(state.optimization ? { ...state.optimization, running: false } : initialOptimizationState)
    setPredictionSettings({ ...defaultPredictionSettings, ...(state.predictionSettings ?? {}) })
    setPredictionOverrides(state.predictionOverrides ?? {})
    setActiveStep(state.activeStep ?? "optimization")
  }

  async function persistSession(document) {
    try {
      const response = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(document),
      })
      if (!response.ok) return
      const saved = await response.json()
      setCurrentSessionId(saved.id)
    } catch {
      /* saving is best-effort; the workbench keeps working offline */
    }
  }

  async function openSession(sessionId) {
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`)
      if (!response.ok) return
      const doc = await response.json()
      applySessionState(doc.state ?? {})
      setCurrentSessionId(doc.id)
      setView("work")
    } catch {
      /* ignore */
    }
  }

  async function removeSession(sessionId) {
    try {
      await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" })
    } catch {
      /* ignore */
    }
    setSessions((current) => (current ?? []).filter((item) => item.id !== sessionId))
    if (sessionId === currentSessionId) setCurrentSessionId(null)
  }

  async function duplicateSessionCard(sessionId) {
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/duplicate`, { method: "POST" })
      if (!response.ok) return
      const refreshed = await fetch(`${API_BASE}/api/sessions`).then((res) => res.json())
      setSessions(refreshed.sessions ?? [])
    } catch {
      /* ignore */
    }
  }

  function startNewCalibration() {
    restoredRef.current = false
    setCurrentSessionId(null)
    setBranches(defaultBranches)
    setSelectedBranchId(defaultBranches[0].id)
    setParameterOverrides({})
    setFittedParameterValues({})
    setSolverSettings(defaultSolverSettings)
    setOptimization(initialOptimizationState)
    setPredictionSettings(defaultPredictionSettings)
    setPredictionOverrides({})
    setBuddayRegion("")
    const preferred = datasets.find((item) => item.author === "Treloar_1944") ?? datasets[0]
    if (preferred) {
      setAuthor(preferred.author)
      setModes(defaultFittingKeys(preferred))
    }
    setActiveStep("experimental")
    setView("work")
  }

  async function startOptimization() {
    if (optimization.running) return
    setFittedParameterValues({})
    setOptimization({
      status: "Running",
      running: true,
      progress: 18,
      initialLoss: 0,
      currentLoss: 0,
      r2: 0.0,
      iterations: 0,
      log: [
        `Started ${solverSettings.method} calibration on backend solver.`,
        `${modes.length} dataset set${modes.length === 1 ? "" : "s"} and ${branches.filter((branch) => branch.enabled).length} active branch${branches.filter((branch) => branch.enabled).length === 1 ? "" : "es"} queued.`,
      ],
    })
    try {
      const response = await fetch(`${API_BASE}/api/calibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildCalibrationPayload()),
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail ?? `Calibration failed with ${response.status}`)
      }
      const result = await response.json()
      const fitted = Object.fromEntries((result.parameters ?? []).map((param) => [param.key, String(Number(param.value).toPrecision(8))]))
      const predictionKeys = preferredPredictionKeys(authorModes, result.modes ?? modes)
      const optimizationSnapshot = {
        status: result.success ? "Converged" : "Finished",
        running: false,
        progress: 100,
        initialLoss: Number(result.initialLoss ?? 0),
        currentLoss: Number(result.loss ?? 0),
        r2: Number(result.r2 ?? 0),
        iterations: Number(result.iterations ?? 0),
        fittedAuthor: result.author,
        fittedModes: result.modes ?? modes,
        prediction: result.prediction,
        log: [
          `${result.success ? "Converged" : "Finished"} after ${result.iterations ?? 0} iterations: loss ${Number(result.loss ?? 0).toExponential(2)}, R2 ${Number(result.r2 ?? 0).toFixed(4)}.`,
          `Fitted ${result.parameters?.length ?? 0} parameter${result.parameters?.length === 1 ? "" : "s"} with ${solverSettings.method}.`,
        ],
      }
      setFittedParameterValues(fitted)
      setPredictionSettings((current) => ({ ...current, modeKeys: predictionKeys }))
      setOptimization(optimizationSnapshot)

      // Archive the finished calibration as a start-page session card.
      const activeBranches = branches.filter((branch) => branch.enabled)
      const modelNames = activeBranches.map((branch) => {
        const model = buildConfiguredModel(modelCatalog.find((item) => item.key === branch.modelKey), branch.modelConfig)
        return model?.name ?? branch.modelKey
      })
      const category = activeBranches.length > 1
        ? "network"
        : (modelCatalog.find((item) => item.key === activeBranches[0]?.modelKey)?.category ?? "phenomenological")
      const sourceName = preview.metadata?.source ?? author
      persistSession({
        id: currentSessionId ?? undefined,
        name: `${sourceName} · ${modelNames.join(" + ")}`,
        summary: {
          author,
          source: sourceName,
          modeShort: preview.metadata?.selectedModeShort ?? "",
          modeCount: modes.length,
          points: preview.metadata?.rows ?? 0,
          models: modelNames,
          category,
          method: solverSettings.method,
          r2: Number(result.r2 ?? 0),
          loss: Number(result.loss ?? 0),
          iterations: Number(result.iterations ?? 0),
          exp: downsamplePoints(preview.series?.[0]?.points ?? preview.points, 20),
          fit: downsamplePoints(result.prediction?.curves?.[0]?.points ?? [], 32),
        },
        state: buildSessionSnapshot({
          activeStep: "optimization",
          fittedParameterValues: fitted,
          optimization: optimizationSnapshot,
          predictionSettings: { ...predictionSettings, modeKeys: predictionKeys },
        }),
      })
    } catch (error) {
      setOptimization((current) => ({
        ...current,
        status: "Error",
        running: false,
        progress: 0,
        log: [`Calibration failed: ${error.message}`, ...current.log].slice(0, 6),
      }))
    }
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
    setFittedParameterValues({})
    setOptimization(initialOptimizationState)
  }

  function updatePredictionSetting(name, value) {
    setPredictionSettings((current) => ({
      ...current,
      [name]: value,
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

  function buildPredictionPayload(modeKeys) {
    const activeBranches = branches.filter((branch) => branch.enabled).map((branch) => {
      const model = buildConfiguredModel(modelCatalog.find((item) => item.key === branch.modelKey), branch.modelConfig)
      const overrides = parameterOverrides[branch.id] ?? {}
      const parameters = Object.fromEntries((model?.parameters ?? []).map((param) => {
        const key = `${branch.id}-${param.name}`
        const value = predictionSettings.manualEdits && predictionOverrides[key] !== "" && predictionOverrides[key] !== undefined
          ? predictionOverrides[key]
          : fittedParameterValues[key] ?? overrides[param.name] ?? param.initial ?? 0
        return [param.name, Number(value)]
      }))
      return {
        id: branch.id,
        name: branch.name,
        modelKey: branch.modelKey,
        modelConfig: model?.config ?? branch.modelConfig ?? {},
        enabled: branch.enabled,
        parameters,
      }
    })
    return {
      author,
      modes: modeKeys,
      branches: activeBranches,
    }
  }

  function togglePredictionMode(modeRecord) {
    setPredictionSettings((current) => {
      const currentKeys = current.modeKeys ?? []
      const active = currentKeys.includes(modeRecord.key)
      const modeKeys = active
        ? currentKeys.length === 1
          ? currentKeys
          : currentKeys.filter((key) => key !== modeRecord.key)
        : [...currentKeys, modeRecord.key]
      return { ...current, modeKeys }
    })
  }

  useEffect(() => {
    const modeKeys = predictionSettings.modeKeys ?? []
    if (!author || !modeKeys.length || !fittedParameterValues || !Object.keys(fittedParameterValues).length) return
    fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPredictionPayload(modeKeys)),
    })
      .then((res) => res.ok ? res.json() : Promise.reject(new Error("Prediction failed")))
      .then((prediction) => {
        setOptimization((current) => ({
          ...current,
          prediction,
        }))
      })
      .catch(() => {})
  }, [author, predictionSettings.modeKeys, predictionSettings.manualEdits, predictionOverrides, fittedParameterValues, branches, modelCatalog, parameterOverrides])

  useEffect(() => {
    if (!optimization.running) return undefined
    const timer = window.setInterval(() => {
      setOptimization((current) => {
        if (!current.running) return current
        const iterations = current.iterations + 1
        const progress = Math.min(95, current.progress + 12)
        if (current.initialLoss <= 0) {
          return {
            ...current,
            progress,
            iterations,
            log: [`Backend solve in progress (${iterations}).`, ...current.log].slice(0, 6),
          }
        }
        const simulatedIterations = current.iterations + 24
        const currentLoss = Number(Math.max(0.0028, current.initialLoss * Math.exp(-progress / 27)).toFixed(6))
        const r2 = Number(Math.min(0.9987, 1 - currentLoss / current.initialLoss).toFixed(4))
        const finished = progress >= 100
        return {
          ...current,
          status: finished ? "Converged" : "Running",
          running: !finished,
          progress,
          iterations: simulatedIterations,
          currentLoss,
          r2,
          log: [
            finished
              ? `Converged after ${simulatedIterations} iterations.`
              : `Iteration ${simulatedIterations}: loss ${currentLoss.toExponential(2)}, R2 ${r2.toFixed(4)}.`,
            ...current.log,
          ].slice(0, 6),
        }
      })
    }, 650)
    return () => window.clearInterval(timer)
  }, [optimization.running])

  if (view === "start") {
    return (
      <StartPage
        sessions={sessions}
        onNew={startNewCalibration}
        onOpen={openSession}
        onDelete={removeSession}
        onDuplicate={duplicateSessionCard}
      />
    )
  }

  return (
    <div className="min-h-screen bg-background text-text-primary">
      <div className="grid min-h-screen grid-cols-[260px_minmax(0,1fr)]">
        <Sidebar activeStep={activeStep} onStepChange={setActiveStep} onHome={() => setView("start")} />
        <div className="flex min-w-0 flex-col">
          <Topbar activeStep={activeStep} />
          <main className="min-w-0 flex-1 overflow-x-hidden p-5 pb-24">
            {activeStep === "experimental" ? (
              <ExperimentalDataPage
                datasets={datasets}
                author={author}
                authorModes={authorModes}
                fittingModes={fittingModes}
                buddayRegions={buddayRegions}
                buddayRegion={buddayRegion}
                modes={modes}
                preview={preview}
                primaryModeMeta={primaryModeMeta}
                onAuthorChange={handleAuthorChange}
                onBuddayRegionChange={handleBuddayRegionChange}
                onModeClick={handleModeClick}
                myDatasets={myDatasets}
                onUploaded={handleUploaded}
                onDeleteUserDataset={handleDeleteUserDataset}
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
                onModelsRefresh={refreshModels}
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
                preview={predictionPreview}
                authorModes={authorModes}
                parameterRows={optimizationParameters}
                optimization={optimization}
                settings={predictionSettings}
                overrides={predictionOverrides}
                onSettingChange={updatePredictionSetting}
                onModeToggle={togglePredictionMode}
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
            onStepChange={setActiveStep}
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

const UPLOAD_FAMILIES = [
  { key: "UT", label: "Uniaxial Tension (UT)", columns: "λ, P", range: [2, 2] },
  { key: "UC", label: "Uniaxial Compression (UC)", columns: "λ, P", range: [2, 2] },
  { key: "ET", label: "Equibiaxial Tension (ET)", columns: "λ, P", range: [2, 2] },
  { key: "PS", label: "Pure Shear (PS)", columns: "λ, P", range: [2, 2] },
  { key: "SS", label: "Simple Shear (SS)", columns: "γ, P₁₂", range: [2, 2] },
  { key: "BT", label: "Biaxial Tension (BT)", columns: "λ₁, λ₂, P₁₁ [, P₂₂]", range: [3, 4] },
]

function parseNumericTable(text) {
  const rows = []
  let skipped = 0
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#") || line.startsWith("%") || line.startsWith("//")) continue
    const values = line.split(/[\s,;\t]+/).filter(Boolean).map(Number)
    if (!values.length || values.some((value) => !Number.isFinite(value))) {
      skipped += 1
      continue
    }
    rows.push(values)
  }
  return { rows, skipped }
}

function uploadValidationError(rows, familyKey) {
  const family = UPLOAD_FAMILIES.find((item) => item.key === familyKey)
  if (!rows.length) return "No numeric rows found in the file."
  if (rows.length < 2) return "At least 2 data points are required."
  const width = rows[0].length
  if (rows.some((row) => row.length !== width)) return "Rows have inconsistent column counts."
  const [min, max] = family.range
  if (width < min || width > max) {
    return `${family.key} expects ${min === max ? min : `${min}–${max}`} columns (${family.columns}), file has ${width}.`
  }
  return null
}

function MyDataPanel({ myDatasets, onUploaded, onDeleteDataset }) {
  const [fileName, setFileName] = useState("")
  const [rows, setRows] = useState([])
  const [skipped, setSkipped] = useState(0)
  const [material, setMaterial] = useState("")
  const [family, setFamily] = useState("UT")
  const [stressType, setStressType] = useState("PK1")
  const [error, setError] = useState("")
  const [saving, setSaving] = useState(false)

  const materials = useMemo(() => [...new Set(myDatasets.map((item) => item.material))], [myDatasets])
  const validation = rows.length ? uploadValidationError(rows, family) : null

  function readFile(file) {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const parsed = parseNumericTable(reader.result)
      setFileName(file.name)
      setRows(parsed.rows)
      setSkipped(parsed.skipped)
      setError(parsed.rows.length ? "" : "No numeric rows found in the file.")
      if (!material && materials.length === 0) setMaterial("My Material")
    }
    reader.readAsText(file)
  }

  async function saveDataset() {
    if (validation || !rows.length || saving) return
    setSaving(true)
    setError("")
    try {
      const response = await fetch(`${API_BASE}/api/user-datasets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          material: material.trim() || "My Material",
          family,
          stressType,
          fileName,
          points: rows,
        }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail ?? `Upload failed (${response.status})`)
      setFileName("")
      setRows([])
      setSkipped(0)
      onUploaded(data.author, data.modeKey)
    } catch (uploadError) {
      setError(uploadError.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mt-3 space-y-3">
      <label
        className="block cursor-pointer rounded-xl border-2 border-dashed border-border-strong bg-subtle px-4 py-5 text-center transition hover:border-primary hover:bg-selection-bg"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          readFile(event.dataTransfer.files?.[0])
        }}
      >
        <input
          type="file"
          accept=".csv,.txt,.tsv,.dat"
          className="hidden"
          onChange={(event) => {
            readFile(event.target.files?.[0])
            event.target.value = ""
          }}
        />
        <div className="text-sm font-semibold text-text-primary">Drop a CSV / TXT file here, or click to browse</div>
        <div className="mt-1 text-xs text-text-muted">
          UT/UC/ET/PS: λ, P · SS: γ, P₁₂ · BT: λ₁, λ₂, P₁₁ [, P₂₂] · units MPa
        </div>
      </label>

      {rows.length > 0 && (
        <div className="rounded-xl border border-border bg-surface p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-text-primary">
            <span className="truncate">{fileName || "pasted data"}</span>
            <span className="text-text-muted">
              {rows.length} rows × {rows[0].length} cols{skipped ? ` · ${skipped} skipped` : ""}
            </span>
          </div>
          <div className="mt-2 overflow-hidden rounded-lg border border-border">
            {rows.slice(0, 4).map((row, index) => (
              <div key={index} className="grid grid-cols-4 gap-1 border-b border-border bg-subtle px-2 py-1 font-mono text-[11px] text-text-muted last:border-b-0">
                {row.slice(0, 4).map((value, column) => (
                  <span key={column}>{Number(value).toPrecision(5)}</span>
                ))}
              </div>
            ))}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2">
            <label className="col-span-2 block text-xs font-bold uppercase text-text-muted">
              Material name
              <input
                className="mt-1 h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm font-normal normal-case outline-none focus:border-primary"
                value={material}
                placeholder="My Material"
                list="mydata-materials"
                onChange={(event) => setMaterial(event.target.value)}
              />
              <datalist id="mydata-materials">
                {materials.map((item) => (
                  <option key={item} value={item} />
                ))}
              </datalist>
            </label>
            <label className="block text-xs font-bold uppercase text-text-muted">
              Loading mode
              <select
                className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm font-normal normal-case outline-none focus:border-primary"
                value={family}
                onChange={(event) => setFamily(event.target.value)}
              >
                {UPLOAD_FAMILIES.map((item) => (
                  <option key={item.key} value={item.key}>{item.label}</option>
                ))}
              </select>
            </label>
            <label className="block text-xs font-bold uppercase text-text-muted">
              Stress type
              <select
                className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm font-normal normal-case outline-none focus:border-primary"
                value={stressType}
                onChange={(event) => setStressType(event.target.value)}
              >
                <option value="PK1">First Piola-Kirchhoff (nominal)</option>
                <option value="cauchy">Cauchy (true)</option>
              </select>
            </label>
          </div>

          {(validation || error) && (
            <p className="mt-2 rounded-lg border border-[#F3C7C7] bg-[#FDF2F2] px-3 py-2 text-xs text-[#B91C1C]">{validation || error}</p>
          )}
          <button
            className="mt-3 w-full rounded-xl primary-gradient px-4 py-2.5 text-sm font-bold text-white shadow-primary-glow transition hover:brightness-105 disabled:opacity-50"
            disabled={Boolean(validation) || saving}
            onClick={saveDataset}
          >
            {saving ? "Saving…" : "Save to My Data"}
          </button>
        </div>
      )}

      {myDatasets.length > 0 && (
        <div>
          <Label>Uploaded tests</Label>
          <div className="flex max-h-44 flex-col gap-1.5 overflow-y-auto pr-1">
            {myDatasets.map((item) => (
              <div key={item.id} className="flex items-center justify-between gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs">
                <span className="min-w-0 flex-1 truncate">
                  <b>{item.material}</b> · {item.modeKey} · {item.points} pts
                  {item.fileName ? <span className="text-text-disabled"> · {item.fileName}</span> : null}
                </span>
                <button
                  className="grid h-6 w-6 shrink-0 place-items-center rounded-md border border-border-strong text-text-muted transition hover:border-[#DC2626] hover:text-[#DC2626]"
                  title="Delete dataset"
                  onClick={() => {
                    if (window.confirm(`Delete "${item.material} · ${item.modeKey}"?`)) onDeleteDataset(item.id)
                  }}
                >
                  <Icon className="text-xs">delete</Icon>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ExperimentalDataPage({
  datasets,
  author,
  authorModes,
  fittingModes,
  buddayRegions,
  buddayRegion,
  modes,
  preview,
  primaryModeMeta,
  onAuthorChange,
  onBuddayRegionChange,
  onModeClick,
  myDatasets,
  onUploaded,
  onDeleteUserDataset,
}) {
  const tensorGroups = groupPreviewSeries(preview, primaryModeMeta)
  const hasDisabledFittingModes = fittingModes.some((option) => fittingDisabledFamilies.has(option.family))
  const isUserAuthor = author.startsWith("user:")
  const [sourceTab, setSourceTab] = useState(isUserAuthor ? "mydata" : "database")
  useEffect(() => {
    setSourceTab(author.startsWith("user:") ? "mydata" : "database")
  }, [author])
  const databaseAuthors = datasets.filter((item) => !item.isUserData)
  const userAuthors = datasets.filter((item) => item.isUserData)
  const tabAuthors = sourceTab === "mydata" ? userAuthors : databaseAuthors

  function switchTab(tab) {
    setSourceTab(tab)
    const pool = tab === "mydata" ? userAuthors : databaseAuthors
    if (pool.length && !pool.some((item) => item.author === author)) {
      onAuthorChange(pool[0].author)
    }
  }

  return (
    <div className="mx-auto grid min-h-[640px] max-w-[1240px] grid-cols-[minmax(360px,0.82fr)_minmax(520px,1.18fr)] gap-4">
      <section className="flex min-w-0 flex-col gap-3">
        <Card title="Data Source">
          <div className="grid grid-cols-2 gap-1 rounded-xl border border-border bg-subtle p-1">
            {[
              ["database", "dns", "Database"],
              ["mydata", "save", "My Data"],
            ].map(([tab, icon, label]) => (
              <button
                key={tab}
                className={`flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                  sourceTab === tab ? "bg-surface text-primary shadow-panel" : "text-text-muted hover:text-text-primary"
                }`}
                onClick={() => switchTab(tab)}
              >
                <Icon className="text-base">{icon}</Icon>
                {label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs leading-5 text-text-muted">
            {sourceTab === "mydata"
              ? "Your uploaded experiments, grouped by material. Same workflow as the built-in database."
              : "Built-in experimental datasets from the literature."}
          </p>
          {tabAuthors.length > 0 && (
            <label className="mt-3 block text-xs font-bold uppercase text-text-muted">
              {sourceTab === "mydata" ? "Material" : "Publication / source"}
              <select className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm normal-case text-text-primary" value={author} onChange={(event) => onAuthorChange(event.target.value)}>
                {tabAuthors.map((item) => (
                  <option key={item.author} value={item.author}>
                    {item.name ?? item.author}
                  </option>
                ))}
              </select>
            </label>
          )}
          {sourceTab === "mydata" && userAuthors.length === 0 && (
            <p className="mt-3 rounded-lg border border-border bg-subtle px-3 py-2 text-xs leading-5 text-text-muted">
              No uploads yet — add your first experiment below.
            </p>
          )}
          {sourceTab === "mydata" && (
            <MyDataPanel myDatasets={myDatasets} onUploaded={onUploaded} onDeleteDataset={onDeleteUserDataset} />
          )}
          {buddayRegions.length > 0 && (
            <label className="mt-3 block text-xs font-bold uppercase text-text-muted">
              Brain region
              <select className="mt-1 w-full rounded-lg border border-border-strong bg-surface px-2 py-2 text-sm normal-case text-text-primary" value={buddayRegion} onChange={(event) => onBuddayRegionChange(event.target.value)}>
                {buddayRegions.map((region) => (
                  <option key={region.key} value={region.key}>
                    {region.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="mt-3">
            <Label>Stress Type</Label>
            <div className="rounded-lg border border-border bg-subtle px-3 py-2 text-sm font-semibold text-text-primary">
              <StressTypeText display={preview.metadata?.stressDisplay} fallback={preview.metadata?.stressType ?? "From dataset"} />
              <span className="ml-2 align-middle text-xs font-normal text-text-muted">read from experimental data</span>
            </div>
          </div>
        </Card>

        <Card title="Fitting Data Sets">
          <p className="mb-2 text-xs leading-5 text-text-muted">
            Select one or more experimental sets to include in the calibration objective. Each database entry is shown separately.
          </p>
          {hasDisabledFittingModes && (
            <p className="mb-2 rounded-lg border border-border bg-subtle px-3 py-2 text-xs leading-5 text-text-muted">
              Biaxial tension (BT) data cannot be used for fitting.
            </p>
          )}
          <div className="flex max-h-72 flex-col gap-2 overflow-y-auto pr-1">
            {fittingModes.map((option) => (
              <ModeButton
                key={option.key}
                option={option}
                meta={modeOptions.find((item) => item.family === option.family) ?? modeOptions[0]}
                active={modes.includes(option.key)}
                onClick={() => onModeClick(option)}
              />
            ))}
          </div>
          {tensorGroups.map((group) => (
            <PreviewTensorOverlay key={group.key} series={group.series} />
          ))}
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
  onModelsRefresh,
}) {
  const modelByKey = useMemo(
    () => Object.fromEntries(models.map((model) => [model.key, model])),
    [models],
  )
  const activeBranches = branches.filter((branch) => branch.enabled)
  const [labBranchId, setLabBranchId] = useState(null)

  async function handleModelSaved(savedKey) {
    await onModelsRefresh()
    if (labBranchId) {
      onUpdateBranch(labBranchId, { modelKey: savedKey, modelConfig: {} })
      onSelectBranch(labBranchId)
    }
    setLabBranchId(null)
  }

  async function handleModelDeleted(modelKey) {
    const modelId = modelKey.replace("user-model:", "")
    try {
      await fetch(`${API_BASE}/api/user-models/${modelId}`, { method: "DELETE" })
    } catch {
      /* ignore */
    }
    const fresh = await onModelsRefresh()
    const freshKeys = new Set((fresh ?? []).map((item) => item.key))
    branches.forEach((branch) => {
      if (!freshKeys.has(branch.modelKey)) {
        const fallback = (fresh ?? []).find((item) => item.key === "NeoHookean") ?? (fresh ?? [])[0]
        if (fallback) onUpdateBranch(branch.id, { modelKey: fallback.key, modelConfig: defaultConfigForModel(fallback) })
      }
    })
  }

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
                baseModel={modelByKey[branch.modelKey]}
                models={models}
                active={branch.id === selectedBranch?.id}
                onSelect={() => onSelectBranch(branch.id)}
                onUpdate={(updates) => onUpdateBranch(branch.id, updates)}
                onToggle={() => onUpdateBranch(branch.id, { enabled: !branch.enabled })}
                onRemove={() => onRemoveBranch(branch.id)}
                onDefineModel={() => setLabBranchId(branch.id)}
              />
            ))}
          </div>
        </Card>

        <ModelLabModal
          open={labBranchId !== null}
          models={models}
          onClose={() => setLabBranchId(null)}
          onSaved={handleModelSaved}
          onDeleted={handleModelDeleted}
        />

        <Card title="Architecture Workspace" className="min-w-0">
          <div className="grid min-w-0 gap-5">
            <section className="min-w-0">
              <h4 className="text-xs font-bold uppercase text-text-muted">Parallel Structure</h4>
              <div className="mt-2 rounded-lg border border-border bg-subtle p-3">
                <ArchitectureVisualizer branches={branches} selectedBranchId={selectedBranch?.id} modelByKey={modelByKey} onSelectBranch={onSelectBranch} />
              </div>
            </section>

            <section className="min-w-0 border-t border-border pt-4">
              <h4 className="mb-2 text-xs font-bold uppercase text-text-muted">Model Equations</h4>
              {selectedModel ? (
                <div className="space-y-3">
                  <FormulaBlock label="Branch Energy Density" value={selectedModel.formula} />
                  {selectedModel.strainFormula && <FormulaBlock label="Generalized Strain" value={selectedModel.strainFormula} />}
                  <ModelReferenceCard model={selectedModel} />
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
            <div className="grid grid-cols-[0.85fr_0.7fr_0.75fr_0.9fr_1fr] rounded-lg border border-border text-sm">
              <div className="contents text-[11px] font-bold uppercase text-text-muted">
                <div className="bg-subtle px-2 py-2">Name</div>
                <div className="bg-subtle px-2 py-2">Initial</div>
                <div className="bg-subtle px-2 py-2">Fitted</div>
                <div className="bg-subtle px-2 py-2">Bounds</div>
                <div className="bg-subtle px-2 py-2">Branch</div>
              </div>
              {parameterRows.slice(0, 12).map((row) => {
                const fitted = row.fitted ?? (optimization.running ? "Solving..." : "")
                return (
                  <div key={`${row.key}-fit`} className="contents">
                    <div className="border-t border-border px-2 py-2 font-semibold"><LatexInline value={parameterSymbol(row.symbol)} fallback={row.symbol} /></div>
                    <div className="border-t border-border px-2 py-2">{row.initial || "-"}</div>
                    <div className="border-t border-border px-2 py-2">{fitted || "-"}</div>
                    <div className="border-t border-border px-2 py-2 text-text-muted">{formatBound(row.lower, "lower")} / {formatBound(row.upper, "upper")}</div>
                    <div className="truncate border-t border-border px-2 py-2 text-text-muted">{row.branch}</div>
                  </div>
                )
              })}
            </div>
            {parameterRows.length > 12 && <p className="mt-2 text-xs text-text-muted">+{parameterRows.length - 12} more parameters included in the solve.</p>}
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
  const fitFactor = 0.82 + Math.min(progress, 100) * 0.0018
  const layout = chartLayout([
    ...allPoints.map((point) => ({ plotX: point.x, plotY: point.y })),
    ...allPoints.map((point) => ({ plotX: point.x, plotY: point.y * fitFactor })),
  ])
  const { width, height, scaleX, scaleY } = layout
  const xSymbol = series[0]?.axisSymbols?.x ?? axisSymbolFromLabel(preview.axes?.x, "\\lambda")
  const ySymbol = series[0]?.axisSymbols?.y ?? axisSymbolFromLabel(preview.axes?.y, "P_{11}")
  const xUnit = axisUnitFromLabel(preview.axes?.x, "-")
  const yUnit = axisUnitFromLabel(preview.axes?.y, "MPa")
  const legendRows = series.map((item, index) => ({
    key: previewSeriesKey(item, index),
    color: colorForSeries(item.modeFamily, index),
    label: formatDisplayLabel(item.modeShortLabel ?? item.modeLabel ?? `Dataset ${index + 1}`),
  }))

  return (
    <div className="flex h-full min-h-[500px] flex-col overflow-hidden rounded-lg border border-border bg-white">
      <div className="relative min-h-[360px] flex-1">
        <svg className="h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Calibration fit chart">
          <rect x="0" y="0" width={width} height={height} fill="#FFFFFF" />
          <ChartGrid layout={layout} />
          {series.map((item, index) => {
            const points = [...(item.points ?? [])]
              .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
              .sort((a, b) => a.x - b.x)
            const color = colorForSeries(item.modeFamily, index)
            const fitPath = points
              .map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y * fitFactor)}`)
              .join(" ")
            return (
              <g key={item.mode ?? item.modeLabel ?? index}>
                {fitPath && <path d={fitPath} fill="none" stroke={color} strokeWidth={chartTheme.fitLineWidth} strokeLinecap="round" strokeLinejoin="round" />}
                {points.map((point, pointIndex) => (
                  <circle key={pointIndex} cx={scaleX(point.x)} cy={scaleY(point.y)} r={chartTheme.pointRadius} fill="#FFFFFF" stroke={color} strokeWidth={chartTheme.pointStrokeWidth} />
                ))}
              </g>
            )
          })}
        </svg>
        <ChartAxisLabels xSymbol={xSymbol} ySymbol={ySymbol} xUnit={xUnit} yUnit={yUnit} />
        <div className="absolute left-3 top-3 max-w-[72%] rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
          <div className="font-semibold">Experimental data and model fit</div>
          <div className="mt-1 truncate text-text-muted">{preview.metadata?.selectedModeShort ?? preview.metadata?.selectedMode ?? "Selected datasets"}</div>
        </div>
      </div>
      <div className="border-t border-border bg-white px-4 py-3 text-xs">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <span className="flex items-center gap-2 font-medium text-text-secondary">
            <span className="h-0.5 w-8 rounded-full bg-text-primary" />
            Model fit
          </span>
          <span className="flex items-center gap-2 font-medium text-text-secondary">
            <span className="h-3 w-3 rounded-full border-2 border-text-primary bg-white" />
            Experimental data
          </span>
          {legendRows.map((item) => (
            <span key={item.key} className="flex min-w-0 items-center gap-2 text-text-muted">
              <span className="h-2.5 w-5 shrink-0 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="truncate">{item.label}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

function chartLayout(points, { yZero = true } = {}) {
  const width = chartTheme.width
  const height = chartTheme.height
  const pad = chartTheme.pad
  const plotWidth = width - pad.left - pad.right
  const plotHeight = height - pad.top - pad.bottom
  const xs = points.map((point) => point.plotX).filter((value) => Number.isFinite(value))
  const ys = points.map((point) => point.plotY).filter((value) => Number.isFinite(value))
  const rawMinX = xs.length ? Math.min(...xs) : 0
  const rawMaxX = xs.length ? Math.max(...xs) : 1
  const rawMinY = ys.length ? Math.min(...ys) : 0
  const rawMaxY = ys.length ? Math.max(...ys) : 1
  const xSpan = Math.max(rawMaxX - rawMinX, 1e-6)
  const yBaseMin = yZero ? Math.min(rawMinY, 0) : rawMinY
  const yBaseMax = yZero ? Math.max(rawMaxY, 0) : rawMaxY
  const ySpan = Math.max(yBaseMax - yBaseMin, 1e-6)
  const xPad = Math.max(xSpan * 0.04, 0.02)
  const yPad = Math.max(ySpan * 0.08, 0.04)
  const xMin = rawMinX >= 0 ? Math.max(0, rawMinX - xPad) : rawMinX - xPad
  const xMax = rawMaxX + xPad
  const yMin = yZero && rawMinY >= 0 ? 0 : yBaseMin - yPad
  const yMax = yBaseMax + yPad
  return {
    width,
    height,
    pad,
    plotWidth,
    plotHeight,
    xMin,
    xMax,
    yMin,
    yMax,
    ticks: chartTheme.ticks,
    scaleX: (x) => pad.left + ((x - xMin) / Math.max(xMax - xMin, 1e-6)) * plotWidth,
    scaleY: (y) => pad.top + ((yMax - y) / Math.max(yMax - yMin, 1e-6)) * plotHeight,
  }
}

function ChartGrid({ layout }) {
  const { width, height, pad, plotWidth, plotHeight, xMin, xMax, yMin, yMax, ticks } = layout
  return (
    <>
      {ticks.map((tick) => {
        const x = pad.left + tick * plotWidth
        const y = pad.top + tick * plotHeight
        const xValue = xMin + tick * (xMax - xMin)
        const yValue = yMax - tick * (yMax - yMin)
        return (
          <g key={tick}>
            <line x1={x} y1={pad.top} x2={x} y2={height - pad.bottom} stroke={chartTheme.grid} strokeDasharray={chartTheme.gridDash} />
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke={chartTheme.grid} strokeDasharray={chartTheme.gridDash} />
            <text x={x} y={height - pad.bottom + 18} textAnchor="middle" fill={chartTheme.tick} fontSize={chartTheme.tickFont}>{formatChartTick(xValue)}</text>
            <text x={pad.left - 12} y={y + 4} textAnchor="end" fill={chartTheme.tick} fontSize={chartTheme.tickFont}>{formatChartTick(yValue)}</text>
          </g>
        )
      })}
      <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} stroke={chartTheme.axis} />
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke={chartTheme.axis} />
    </>
  )
}

function ChartAxisLabels({ xSymbol, ySymbol, xUnit, yUnit }) {
  const xFormula = axisLabelFormula(xSymbol, xUnit)
  const yFormula = axisLabelFormula(ySymbol, yUnit)
  return (
    <>
      <div
        className="chart-axis-label pointer-events-none absolute bottom-2 flex items-center justify-center text-sm"
        style={{ left: chartTheme.pad.left, right: chartTheme.pad.right }}
      >
        <LatexInline value={xFormula} fallback={`${plainAxisLabel(xSymbol)} ${xUnit ?? ""}`} />
      </div>
      <div
        className="chart-axis-label pointer-events-none absolute flex items-center justify-center text-sm"
        style={{ left: 10, top: chartTheme.pad.top, bottom: chartTheme.pad.bottom, width: 28 }}
      >
        <div className="origin-center -rotate-90 whitespace-nowrap">
          <LatexInline value={yFormula} fallback={`${plainAxisLabel(ySymbol)} ${yUnit ?? ""}`} />
        </div>
      </div>
    </>
  )
}

function axisLabelFormula(symbol, unit) {
  const latexUnit = latexAxisUnit(unit)
  return latexUnit ? `${symbol}\\,${latexUnit}` : symbol
}

function latexAxisUnit(unit) {
  const inner = String(unit ?? "").replace(/^\[/, "").replace(/\]$/, "").trim()
  if (!inner) return ""
  if (inner === "-") return "[-]"
  return `[\\mathrm{${inner.replace(/[{}\\]/g, "")}}]`
}

function axisUnitFromLabel(label, fallback = "") {
  const text = String(label ?? "")
  const bracket = text.match(/\[([^\]]+)\]/)
  const paren = text.match(/\(([^)]+)\)\s*$/)
  const unit = bracket?.[1] ?? paren?.[1] ?? fallback
  if (!unit) return ""
  return `[${unit}]`
}

function axisSymbolFromLabel(label, fallback) {
  const text = String(label ?? "")
  if (/sigma|σ/.test(text)) {
    if (/22|₂₂/.test(text)) return "\\sigma_{22}"
    if (/12|₁₂/.test(text)) return "\\sigma_{12}"
    return "\\sigma_{11}"
  }
  if (/(^|[^A-Za-z])P([^A-Za-z]|$)|P[₁₂0-9_]/.test(text)) {
    if (/22|₂₂/.test(text)) return "P_{22}"
    if (/12|₁₂/.test(text)) return "P_{12}"
    return "P_{11}"
  }
  if (/lambda_1|λ₁/.test(text)) return "\\lambda_1"
  if (/lambda|λ/.test(text)) return "\\lambda"
  if (/gamma|γ/.test(text)) return "\\gamma"
  return fallback
}

function PredictionPage({
  preview,
  authorModes,
  parameterRows,
  optimization,
  settings,
  overrides,
  onSettingChange,
  onModeToggle,
  onOverrideChange,
  onResetOverrides,
}) {
  const changedCount = Object.keys(overrides).length
  const selectedModeKeys = settings.modeKeys ?? []
  const prediction = useMemo(
    () => buildPredictionSeries(preview, parameterRows, settings, overrides, optimization, authorModes),
    [preview, parameterRows, settings, overrides, optimization, authorModes],
  )
  const allPredictionPoints = [
    ...prediction.experimental,
    ...prediction.curves.flatMap((curve) => curve.points),
  ]
  const xs = allPredictionPoints.map((point) => point.x)
  const peakStress = allPredictionPoints.reduce((max, point) => Math.max(max, point.y, point.y2 ?? 0), 0)
  const stretchRange = xs.length ? `${Math.min(...xs).toFixed(2)}-${Math.max(...xs).toFixed(2)}` : "-"
  const dataPointCount = prediction.experimental.length
  const stressOutput = stressPlainText(preview.metadata?.stressDisplay, preview.metadata?.stressType ?? "From dataset")
  const metrics = [
    ["Peak stress", `${peakStress.toFixed(3)} MPa`],
    ["Data range", stretchRange],
    ["Data points", dataPointCount],
    ["Stress output", stressOutput],
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
          <Card title="Prediction Setup">
            <div>
              <Label>Loading Modes</Label>
              <div className="max-h-48 space-y-2 overflow-y-auto pr-1">
                {authorModes.map((option) => {
                  const active = selectedModeKeys.includes(option.key)
                  const meta = modeOptions.find((item) => item.family === option.family) ?? modeOptions[0]
                  const label = option.shortLabel ?? option.label
                  return (
                    <button
                      key={option.key}
                      className={`flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left ${active ? "border-primary bg-selection-bg" : "border-border-strong hover:bg-subtle"}`}
                      onClick={() => onModeToggle(option)}
                    >
                      <span className={`h-3 w-3 rounded-full ${meta.dot}`} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-semibold">{formatDisplayLabel(label)}</span>
                        <span className="block truncate text-xs text-text-muted">{modeFamilyName(option.family)} · {option.points} pts · <StressText display={option.stressDisplay} fallback={option.stressType} /></span>
                      </span>
                      <span className={`grid h-5 w-5 place-items-center rounded border ${active ? "border-primary bg-primary text-white" : "border-border-strong bg-surface"}`}>
                        {active && <Icon className="text-sm">check_circle</Icon>}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-border bg-subtle p-2">
                <Label>Data Points</Label>
                <div className="text-sm font-semibold">{dataPointCount}</div>
              </div>
              <div className="rounded-lg border border-border bg-subtle p-2">
                <Label>Stress Output</Label>
                <div className="text-sm font-semibold">{stressOutput}</div>
              </div>
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
                <span>Source</span>
                <span>Manual</span>
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
              <PredictionChart prediction={prediction} preview={preview} />
            </div>
          </Card>
        </div>
      </section>
    </div>
  )
}

function PredictionChart({ prediction, preview }) {
  const hasSecondary = prediction.experimental.some((point) => point.x2 !== undefined && point.y2 !== undefined)
    || prediction.curves.some((curve) => (curve.points ?? []).some((point) => point.x2 !== undefined && point.y2 !== undefined))
  const primarySymbols = preview.series?.[0]?.axisSymbols ?? {}
  const xSymbol = primarySymbols.x ?? axisSymbolFromLabel(preview.axes?.x, "\\lambda")
  const ySymbol = primarySymbols.y ?? axisSymbolFromLabel(preview.axes?.y, "P_{11}")
  const xUnit = axisUnitFromLabel(preview.axes?.x, "-")
  const yUnit = axisUnitFromLabel(preview.axes?.y, "MPa")
  const primaryExperimental = prediction.experimental.map((point) => ({ ...point, plotX: point.x, plotY: point.y }))
  const primaryCurves = prediction.curves.map((curve) => ({
    ...curve,
    points: (curve.points ?? []).map((point) => ({ ...point, plotX: point.x, plotY: point.y })),
  }))
  const secondaryExperimental = prediction.experimental
    .filter((point) => point.x2 !== undefined && point.y2 !== undefined)
    .map((point) => ({ ...point, plotX: point.x, plotY: point.y2 }))
  const secondaryCurves = prediction.curves.map((curve) => ({
    ...curve,
    points: (curve.points ?? [])
      .filter((point) => point.x2 !== undefined && point.y2 !== undefined)
      .map((point) => ({ ...point, plotX: point.x, plotY: point.y2 })),
  }))
  return (
    <div className={hasSecondary ? "grid h-full min-h-[640px] gap-3" : "h-full min-h-[540px]"}>
      <StressComponentChart
        title={hasSecondary ? "Component P11" : "Prediction review"}
        detail={preview.metadata?.selectedModeShort ?? preview.metadata?.selectedMode ?? "Selected datasets"}
        experimental={primaryExperimental}
        curves={primaryCurves}
        xLabel={preview.axes?.x ?? "Stretch lambda_1 (-)"}
        yLabel={preview.axes?.y ?? "Nominal stress P11"}
        xSymbol={xSymbol}
        ySymbol={ySymbol}
        xUnit={xUnit}
        yUnit={yUnit}
        minHeight={hasSecondary ? 420 : 540}
      />
      {hasSecondary && (
        <StressComponentChart
          title="Component P22"
          detail={secondaryDetail(prediction.curves)}
          experimental={secondaryExperimental}
          curves={secondaryCurves}
          xLabel={preview.axes?.x ?? "Stretch lambda_1 (-)"}
          yLabel={secondaryStressLabel(preview)}
          xSymbol={xSymbol}
          ySymbol={preview.metadata?.stressType === "Cauchy" ? "\\sigma_{22}" : "P_{22}"}
          xUnit={xUnit}
          yUnit={yUnit}
          minHeight={320}
        />
      )}
    </div>
  )
}

function StressComponentChart({ title, detail, experimental, curves, xLabel, yLabel, xSymbol, ySymbol, xUnit, yUnit, minHeight = 460 }) {
  const curvePoints = curves.flatMap((curve) => curve.points ?? [])
  const points = [...experimental, ...curvePoints]
  const layout = chartLayout(points)
  const { width, height, scaleX, scaleY } = layout
  const axisX = xSymbol ?? axisSymbolFromLabel(xLabel, "\\lambda")
  const axisY = ySymbol ?? axisSymbolFromLabel(yLabel, "P_{11}")
  const unitX = xUnit ?? axisUnitFromLabel(xLabel, "-")
  const unitY = yUnit ?? axisUnitFromLabel(yLabel, "MPa")
  const legendColor = colorForSeries(curves[0]?.family ?? experimental[0]?.family, 0)
  const legendItems = curves.length
    ? curves.map((curve, index) => ({
      key: curve.key ?? `${curve.family}-${index}`,
      color: colorForSeries(curve.family, index),
      label: curve.label ?? curve.name ?? `Curve ${index + 1}`,
    }))
    : [{ key: "experimental", color: legendColor, label: "Experimental data" }]

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-border bg-white" style={{ minHeight }}>
      <div className="relative min-h-[360px] flex-1">
        <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} chart`}>
          <rect width={width} height={height} fill="#FFFFFF" />
          <ChartGrid layout={layout} />
          {curves.map((curve, index) => {
            const curvePath = (curve.points ?? []).map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${scaleX(point.plotX)} ${scaleY(point.plotY)}`).join(" ")
            return curvePath ? <path key={curve.key} d={curvePath} fill="none" stroke={colorForSeries(curve.family, index)} strokeWidth={chartTheme.fitLineWidth} strokeLinecap="round" strokeLinejoin="round" /> : null
          })}
          {experimental.map((point, index) => (
            <circle key={`${point.seriesKey}-${point.plotX}-${point.plotY}-${index}`} cx={scaleX(point.plotX)} cy={scaleY(point.plotY)} r={chartTheme.pointRadius} fill="#FFFFFF" stroke={colorForSeries(point.family, point.seriesIndex ?? 0)} strokeWidth={chartTheme.pointStrokeWidth} />
          ))}
        </svg>
        <ChartAxisLabels xSymbol={axisX} ySymbol={axisY} xUnit={unitX} yUnit={unitY} />
        <div className="absolute left-3 top-3 max-w-[70%] rounded-md border border-border-strong bg-white/95 px-2 py-1 text-xs shadow-panel">
          <div className="font-semibold">{title}</div>
          <div className="mt-1 truncate text-text-muted">{detail}</div>
        </div>
      </div>
      <div className="border-t border-border bg-white px-4 py-3 text-xs">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <span className="flex items-center gap-2 font-medium text-text-secondary">
            <span className="h-0.5 w-8 rounded-full bg-text-primary" />
            Prediction curve
          </span>
          <span className="flex items-center gap-2 font-medium text-text-secondary">
            <span className="h-3 w-3 rounded-full border-2 border-text-primary bg-white" />
            Experimental data
          </span>
          {legendItems.map((item) => (
            <span key={item.key} className="flex min-w-0 items-center gap-2 text-text-muted">
              <span className="h-2.5 w-5 shrink-0 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="truncate">{formatDisplayLabel(item.label)}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

function secondaryStressLabel(preview) {
  return preview.metadata?.stressType === "Cauchy" ? "Cauchy stress σ₂₂" : "Nominal stress P₂₂"
}

function secondaryDetail(curves) {
  const fixed = curves
    .filter((curve) => curve.fixedStretch !== null && curve.fixedStretch !== undefined)
    .slice(0, 2)
    .map((curve) => `${curve.fixedStretchLabel ?? "fixed"}=${Number(curve.fixedStretch).toFixed(2)}`)
    .join(" · ")
  return fixed || "Second biaxial component"
}

function buildPredictionSeries(preview, parameterRows, settings, overrides, optimization, authorModes) {
  const series = preview.series?.length
    ? preview.series
    : [{ modeFamily: "UT", modeLabel: "Experimental", points: preview.points ?? [] }]
  const fittedModes = new Set(optimization.fittedAuthor === preview.author ? optimization.fittedModes ?? [] : [])
  const experimental = series
    .filter((item) => !fittedModes.has(item.mode))
    .flatMap((item, seriesIndex) => (item.points ?? []).map((point) => ({
    ...point,
    family: item.modeFamily,
    label: item.modeLabel,
    fixedStretch: item.fixedStretch,
    fixedStretchLabel: item.fixedStretchLabel,
    seriesIndex,
    seriesKey: item.mode,
  })))
  const backendCurveByKey = new Map((optimization.prediction?.curves ?? []).map((curve) => [curve.key, curve]))
  if (optimization.prediction?.author === preview.author && series.length && series.every((item) => backendCurveByKey.has(item.mode))) {
    return {
      experimental,
      curves: series.map((item, seriesIndex) => {
        const curve = backendCurveByKey.get(item.mode)
        return {
          key: curve.key ?? item.mode ?? `${item.modeFamily}-${seriesIndex}`,
          family: curve.family ?? item.modeFamily,
          label: curve.label ?? item.modeLabel,
          fixedStretch: curve.fixedStretch ?? item.fixedStretch,
          fixedStretchLabel: curve.fixedStretchLabel ?? item.fixedStretchLabel,
          points: curve.points ?? [],
        }
      }),
    }
  }
  const values = parameterRows.map((row, index) => {
    const override = overrides[row.key]
    const value = settings.manualEdits && override !== "" && override !== undefined ? override : row.value
    const numeric = Number(value)
    return Number.isFinite(numeric) ? Math.abs(numeric) / (index + 3) : 0
  })
  const parameterScale = values.reduce((sum, value) => sum + value, 0)
  const sourceBoost = settings.manualEdits ? 1.08 : 1 + optimization.progress / 700
  const parameterEffect = Math.max(0.75, Math.min(1.35, (0.92 + parameterScale * 0.01) * sourceBoost))
  const modeFactors = {
    CSS: 0.88,
    UT: 1,
    UC: 0.92,
    PS: 1.16,
    ET: 1.34,
    SS: 0.76,
    BT: 1.24,
  }
  const familyNormalizer = (settings.modeKeys ?? [])
    .map((key) => authorModes.find((item) => item.key === key)?.family)
    .filter(Boolean)
    .map((family) => modeFactors[family] ?? 1)
  const averageFamilyFactor = familyNormalizer.length
    ? familyNormalizer.reduce((sum, value) => sum + value, 0) / familyNormalizer.length
    : 1
  const curves = series.map((item, seriesIndex) => {
    const sourcePoints = (item.points ?? [])
      .map((point) => ({ x: Number(point.x), y: Number(point.y), x2: Number(point.x2), y2: Number(point.y2) }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
      .sort((a, b) => a.x - b.x)
    const secondarySourcePoints = sourcePoints
      .filter((point) => Number.isFinite(point.x2) && Number.isFinite(point.y2))
      .map((point) => ({ x: point.x2, y: point.y2 }))
      .sort((a, b) => a.x - b.x)
    const familyFactor = (modeFactors[item.modeFamily] ?? 1) / averageFamilyFactor
    const primaryPoints = buildSmoothPredictionCurve(sourcePoints, parameterEffect * familyFactor)
    const secondaryPoints = buildSmoothPredictionCurve(secondarySourcePoints, parameterEffect * familyFactor)
    return {
      key: item.mode ?? `${item.modeFamily}-${seriesIndex}`,
      family: item.modeFamily,
      label: item.modeLabel,
      fixedStretch: item.fixedStretch,
      fixedStretchLabel: item.fixedStretchLabel,
      points: primaryPoints.map((point, index) => ({
        ...point,
        ...(secondaryPoints[index] ? { x2: secondaryPoints[index].x, y2: secondaryPoints[index].y } : {}),
      })),
    }
  })
  return { experimental, curves }
}

function buildSmoothPredictionCurve(points, scale = 1) {
  if (!points.length) {
    return []
  }
  if (points.length === 1) {
    return [{ ...points[0], y: points[0].y * scale }]
  }
  const minX = points[0].x
  const maxX = points[points.length - 1].x
  const sampleCount = Math.max(32, Math.min(120, points.length * 10))
  const quadratic = fitQuadratic(points)
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length
  return Array.from({ length: sampleCount }, (_, index) => {
    const t = sampleCount === 1 ? 0 : index / (sampleCount - 1)
    const x = minX + t * (maxX - minX)
    const rawY = quadratic
      ? quadratic[0] + quadratic[1] * x + quadratic[2] * x * x
      : interpolateY(points, x)
    return { x, y: meanY + (rawY - meanY) * scale }
  })
}

function interpolateY(points, x) {
  if (x <= points[0].x) return points[0].y
  for (let index = 1; index < points.length; index += 1) {
    const left = points[index - 1]
    const right = points[index]
    if (x <= right.x) {
      const t = (x - left.x) / Math.max(right.x - left.x, 1e-9)
      return left.y + t * (right.y - left.y)
    }
  }
  return points[points.length - 1].y
}

function fitQuadratic(points) {
  if (points.length < 3) return null
  const sums = points.reduce((acc, point) => {
    const x2 = point.x * point.x
    acc.x += point.x
    acc.x2 += x2
    acc.x3 += x2 * point.x
    acc.x4 += x2 * x2
    acc.y += point.y
    acc.xy += point.x * point.y
    acc.x2y += x2 * point.y
    return acc
  }, { x: 0, x2: 0, x3: 0, x4: 0, y: 0, xy: 0, x2y: 0 })
  return solveLinearSystem3(
    [
      [points.length, sums.x, sums.x2],
      [sums.x, sums.x2, sums.x3],
      [sums.x2, sums.x3, sums.x4],
    ],
    [sums.y, sums.xy, sums.x2y],
  )
}

function solveLinearSystem3(matrix, vector) {
  const a = matrix.map((row, index) => [...row, vector[index]])
  for (let pivot = 0; pivot < 3; pivot += 1) {
    let best = pivot
    for (let row = pivot + 1; row < 3; row += 1) {
      if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row
    }
    if (Math.abs(a[best][pivot]) < 1e-9) return null
    if (best !== pivot) {
      const current = a[pivot]
      a[pivot] = a[best]
      a[best] = current
    }
    const divisor = a[pivot][pivot]
    for (let col = pivot; col < 4; col += 1) a[pivot][col] /= divisor
    for (let row = 0; row < 3; row += 1) {
      if (row === pivot) continue
      const factor = a[row][pivot]
      for (let col = pivot; col < 4; col += 1) a[row][col] -= factor * a[pivot][col]
    }
  }
  return [a[0][3], a[1][3], a[2][3]]
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

const MODEL_LAB_HINT = "Symbols: I1, I2 (invariant) or lambda_1..3 (stretch) · functions: exp, log, sqrt, sinh, cosh, tanh, atan… · everything else becomes a parameter"

function ModelLabModal({ open, models, onClose, onSaved, onDeleted }) {
  const [name, setName] = useState("")
  const [kind, setKind] = useState("invariant")
  const [expression, setExpression] = useState("")
  const [notes, setNotes] = useState("")
  const [paramValues, setParamValues] = useState({})
  const [check, setCheck] = useState(null) // {latex, parameters, error}
  const [feedback, setFeedback] = useState(null) // {errors, warnings}
  const [saving, setSaving] = useState(false)

  const userModels = models.filter((item) => item.isUserModel)

  useEffect(() => {
    if (!open) return undefined
    if (!expression.trim()) {
      setCheck(null)
      return undefined
    }
    let alive = true
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/user-models/validate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind, expression }),
        })
        const data = await response.json().catch(() => ({}))
        if (!alive) return
        if (!response.ok) {
          setCheck({ error: data.detail ?? "Could not parse the expression." })
        } else {
          setCheck({ latex: data.latex, parameters: data.parameters ?? [] })
        }
      } catch {
        if (alive) setCheck({ error: "Validation request failed." })
      }
    }, 450)
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [open, kind, expression])

  if (!open) return null

  const detected = check?.parameters ?? []
  const latexHtml = check?.latex ? renderFormula(check.latex) : ""

  function paramField(paramName, field, fallback = "") {
    return paramValues[paramName]?.[field] ?? fallback
  }

  function setParamField(paramName, field, value) {
    setParamValues((current) => ({
      ...current,
      [paramName]: { ...current[paramName], [field]: value },
    }))
  }

  async function save() {
    if (saving) return
    setSaving(true)
    setFeedback(null)
    try {
      const payload = {
        name,
        kind,
        expression,
        notes,
        params: detected.map((paramName) => ({
          name: paramName,
          initial: Number(paramField(paramName, "initial", "1")) || 1,
          lower: paramField(paramName, "lower") === "" ? null : Number(paramField(paramName, "lower")),
          upper: paramField(paramName, "upper") === "" ? null : Number(paramField(paramName, "upper")),
        })),
      }
      const response = await fetch(`${API_BASE}/api/user-models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        setFeedback({ errors: [data.detail ?? `Save failed (${response.status})`], warnings: [] })
        return
      }
      setFeedback({ errors: [], warnings: data.warnings ?? [] })
      setName("")
      setExpression("")
      setNotes("")
      setParamValues({})
      setCheck(null)
      onSaved(data.key, data.warnings ?? [])
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#0E1B33]/45 p-6 backdrop-blur-sm" onClick={onClose}>
      <div
        className="max-h-[88vh] w-full max-w-[760px] overflow-y-auto rounded-2xl border border-border bg-surface p-5 shadow-float"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-[16px] font-semibold tracking-[-0.01em]">
            <span className="h-3.5 w-1 rounded-full primary-gradient" />
            Define Custom Model
          </h3>
          <button className="grid h-8 w-8 place-items-center rounded-lg border border-border-strong text-text-muted hover:bg-subtle" onClick={onClose}>✕</button>
        </div>

        <div className="grid grid-cols-[1fr_auto] gap-3">
          <label className="block text-xs font-bold uppercase text-text-muted">
            Model name
            <input
              className="mt-1 h-10 w-full rounded-lg border border-border-strong bg-surface px-2.5 text-sm font-normal normal-case outline-none focus:border-primary"
              value={name}
              placeholder="e.g. Gent"
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <div className="block text-xs font-bold uppercase text-text-muted">
            Formulation
            <div className="mt-1 grid grid-cols-2 gap-1 rounded-xl border border-border bg-subtle p-1">
              {[
                ["invariant", "Ψ(I₁, I₂)"],
                ["stretch", "Ψ(λ₁, λ₂, λ₃)"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  className={`rounded-lg px-3 py-1.5 text-sm font-semibold normal-case transition ${
                    kind === value ? "bg-surface text-primary shadow-panel" : "text-text-muted hover:text-text-primary"
                  }`}
                  onClick={() => setKind(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <label className="mt-3 block text-xs font-bold uppercase text-text-muted">
          Strain-energy expression Ψ
          <textarea
            className="mt-1 min-h-[76px] w-full rounded-lg border border-border-strong bg-surface px-2.5 py-2 font-mono text-[13px] font-normal normal-case leading-relaxed outline-none focus:border-primary"
            value={expression}
            placeholder={kind === "invariant" ? "-(mu*Jm/2) * log(1 - (I1-3)/Jm)" : "(mu/alpha)*(lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)"}
            spellCheck={false}
            onChange={(event) => setExpression(event.target.value)}
          />
        </label>
        <p className="mt-1 text-[11px] leading-4 text-text-disabled">{MODEL_LAB_HINT}</p>

        {check?.error && (
          <p className="mt-2 rounded-lg border border-[#F3C7C7] bg-[#FDF2F2] px-3 py-2 text-xs text-[#B91C1C]">{check.error}</p>
        )}
        {latexHtml && (
          <div className="formula-block mt-2 overflow-x-auto rounded-lg border border-border bg-subtle px-4 py-3" dangerouslySetInnerHTML={{ __html: latexHtml }} />
        )}

        {detected.length > 0 && (
          <div className="mt-3">
            <Label>Parameters (auto-detected)</Label>
            <div className="overflow-hidden rounded-lg border border-border">
              <div className="grid grid-cols-[1fr_1.1fr_1fr_1fr] border-b border-border bg-subtle px-3 py-1.5 text-[11px] font-bold uppercase text-text-muted">
                <span>Name</span><span>Initial</span><span>Lower</span><span>Upper</span>
              </div>
              {detected.map((paramName) => (
                <div key={paramName} className="grid grid-cols-[1fr_1.1fr_1fr_1fr] items-center gap-2 border-b border-border px-3 py-1.5 text-sm last:border-b-0">
                  <span className="font-semibold"><LatexInline value={parameterSymbol(paramName)} fallback={paramName} /></span>
                  {["initial", "lower", "upper"].map((field) => (
                    <input
                      key={field}
                      className="h-8 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary"
                      type="number"
                      step="any"
                      placeholder={field === "initial" ? "1" : "—"}
                      value={paramField(paramName, field)}
                      onChange={(event) => setParamField(paramName, field, event.target.value)}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        <label className="mt-3 block text-xs font-bold uppercase text-text-muted">
          Notes / reference (optional)
          <input
            className="mt-1 h-9 w-full rounded-lg border border-border-strong bg-surface px-2.5 text-sm font-normal normal-case outline-none focus:border-primary"
            value={notes}
            placeholder="e.g. Gent (1996), Rubber Chem. Technol."
            onChange={(event) => setNotes(event.target.value)}
          />
        </label>

        {feedback?.errors?.length > 0 && (
          <p className="mt-3 rounded-lg border border-[#F3C7C7] bg-[#FDF2F2] px-3 py-2 text-xs text-[#B91C1C]">{feedback.errors.join(" ")}</p>
        )}
        {feedback?.warnings?.length > 0 && (
          <p className="mt-3 rounded-lg border border-[#F3DFBB] bg-[#FDF8EC] px-3 py-2 text-xs text-[#92600A]">{feedback.warnings.join(" ")}</p>
        )}

        <div className="mt-4 flex items-center justify-end gap-2">
          <button className="rounded-xl border border-border-strong bg-surface px-4 py-2 text-sm font-semibold hover:bg-subtle" onClick={onClose}>Cancel</button>
          <button
            className="rounded-xl primary-gradient px-5 py-2 text-sm font-bold text-white shadow-primary-glow transition hover:brightness-105 disabled:opacity-50"
            disabled={!name.trim() || !expression.trim() || Boolean(check?.error) || !detected.length || saving}
            onClick={save}
          >
            {saving ? "Checking & saving…" : "Save Model"}
          </button>
        </div>

        {userModels.length > 0 && (
          <div className="mt-5 border-t border-dashed border-border pt-3">
            <Label>My models</Label>
            <div className="flex max-h-44 flex-col gap-1.5 overflow-y-auto pr-1">
              {userModels.map((item) => (
                <div key={item.key} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-1.5">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{item.name}</div>
                    <div className="formula-compact truncate" dangerouslySetInnerHTML={{ __html: renderFormula(item.formula, false) }} />
                  </div>
                  <button
                    className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-border-strong text-text-muted transition hover:border-[#DC2626] hover:text-[#DC2626]"
                    title="Delete model"
                    onClick={() => {
                      if (window.confirm(`Delete model "${item.name}"?`)) onDeleted(item.key)
                    }}
                  >
                    <Icon className="text-xs">delete</Icon>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function BranchCard({ branch, total, model, baseModel, models, active, onSelect, onUpdate, onToggle, onRemove, onDefineModel }) {
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
      <label className="mt-3 block" onClick={(event) => event.stopPropagation()}>
        <Label>Constitutive Model</Label>
        <select
          className="h-9 w-full rounded-lg border border-border-strong bg-surface px-2 text-sm outline-none focus:border-primary"
          value={branch.modelKey}
          onChange={(event) => {
            const nextModel = models.find((item) => item.key === event.target.value)
            onUpdate({
              modelKey: event.target.value,
              modelConfig: defaultConfigForModel(nextModel),
            })
          }}
        >
          <optgroup label="Built-in">
            {models.filter((item) => !item.isUserModel).map((item) => (
              <option key={item.key} value={item.key}>{item.name}</option>
            ))}
          </optgroup>
          {models.some((item) => item.isUserModel) && (
            <optgroup label="My Models">
              {models.filter((item) => item.isUserModel).map((item) => (
                <option key={item.key} value={item.key}>{item.name}</option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          className="mt-1.5 flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
          onClick={onDefineModel}
        >
          <Icon className="text-sm">add</Icon>
          Define custom model…
        </button>
      </label>
      <div className="mt-3" onClick={(event) => event.stopPropagation()}>
        <ModelConfigurationPanel
          model={baseModel}
          config={branch.modelConfig}
          onChange={(modelConfig) => onUpdate({ modelConfig })}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-1">
        <IconButton label={branch.enabled ? "Disable" : "Enable"} onClick={onToggle}>power_settings_new</IconButton>
        <IconButton label="Remove" disabled={total === 1} onClick={onRemove}>delete</IconButton>
      </div>
    </div>
  )
}

function ModelReferenceCard({ model }) {
  if (!model) return null
  return (
    <div className="rounded-lg border border-border bg-subtle p-3">
      <h3 className="text-sm font-semibold">{model.name}</h3>
      <p className="mt-1 text-xs text-text-muted">{typeLabel(model.type)} · {model.category}{model.detail ? ` · ${model.detail}` : ""}</p>
      {model.reference && (
        <p className="mt-2 text-xs text-text-muted">
          Reference:{" "}
          {model.referenceUrl ? (
            <a className="font-semibold text-primary hover:underline" href={model.referenceUrl} target="_blank" rel="noreferrer">
              {model.reference}
            </a>
          ) : model.reference}
        </p>
      )}
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
        <line x1={railLeft} y1={top - 24} x2={railLeft} y2={top + (branches.length - 1) * rowGap + 24} stroke="#8494AC" strokeWidth="4" strokeLinecap="round" />
        <line x1={railRight} y1={top - 24} x2={railRight} y2={top + (branches.length - 1) * rowGap + 24} stroke="#8494AC" strokeWidth="4" strokeLinecap="round" />
        <text x={railLeft} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#5B6B84">F input</text>
        <text x={railRight} y={top - 38} textAnchor="middle" fontSize="12" fontWeight="700" fill="#5B6B84">P total</text>
        {branches.map((branch, index) => {
          const model = buildConfiguredModel(modelByKey[branch.modelKey], branch.modelConfig)
          const y = top + index * rowGap
          const selected = branch.id === selectedBranchId
          const color = selected ? "#2563EB" : branch.enabled ? "#3B4A66" : "#C0CADC"
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
              <rect x={springStart + 112} y={y - 25} width="210" height="50" rx="8" fill={selected ? "#ECF2FE" : "#F6F9FF"} stroke={selected ? "#2563EB" : "#E3E9F2"} />
              <text x={springStart + 126} y={y - 5} fontSize="12" fontWeight="700" fill="#0E1B33">{branch.name}</text>
              <text x={springStart + 126} y={y + 13} fontSize="11" fill="#5B6B84">{model?.name ?? branch.modelKey}{model?.detail ? ` · ${model.detail}` : ""}</text>
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

function BrandMark({ className = "h-8 w-8" }) {
  return (
    <svg className={className} viewBox="0 0 64 64" role="img" aria-label="Calibration for Hyperelasticity logo">
      <defs>
        <linearGradient id="bm-tile" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#4C8DFF" />
          <stop offset="0.5" stopColor="#2563EB" />
          <stop offset="1" stopColor="#1B3A96" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="15" fill="url(#bm-tile)" />
      <g stroke="#FFFFFF" strokeOpacity="0.16" strokeWidth="1">
        <path d="M27 17V47" />
        <path d="M40 17V47" />
        <path d="M16 30H50" />
      </g>
      <path d="M16 47V15" fill="none" stroke="#FFFFFF" strokeOpacity="0.5" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 47H50" fill="none" stroke="#FFFFFF" strokeOpacity="0.5" strokeWidth="2" strokeLinecap="round" />
      <path d="M18 44C28 43 34 40 39 33C43.5 27 46 22 48 16" fill="none" stroke="#FFFFFF" strokeWidth="4.2" strokeLinecap="round" strokeLinejoin="round" />
      {[["24", "43"], ["34", "37"], ["44", "24"]].map(([cx, cy]) => (
        <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="2.6" fill="#FBBF24" stroke="#FFFFFF" strokeWidth="1.4" />
      ))}
    </svg>
  )
}

function SessionSparkline({ summary }) {
  const exp = summary?.exp ?? []
  const fit = summary?.fit ?? []
  const all = [...exp, ...fit]
  if (all.length < 2) {
    return <div className="grid h-full place-items-center text-[11px] text-text-disabled">No curve</div>
  }
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const sx = (x) => 10 + ((x - xMin) / (xMax - xMin || 1)) * 130
  const sy = (y) => 56 - ((y - yMin) / (yMax - yMin || 1)) * 48
  const fitPath = fit.map((p, i) => `${i ? "L" : "M"} ${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ")
  const dotStep = Math.max(1, Math.floor(exp.length / 5))
  const dots = exp.filter((_, i) => i % dotStep === 0).slice(0, 6)
  return (
    <svg viewBox="0 0 150 64" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
      {fitPath && <path d={fitPath} fill="none" stroke="#2563EB" strokeWidth="2.4" strokeLinecap="round" />}
      <g fill="#fff" stroke="#DC2626" strokeWidth="1.6">
        {dots.map((p, i) => (
          <circle key={i} cx={sx(p[0]).toFixed(1)} cy={sy(p[1]).toFixed(1)} r="2.6" />
        ))}
      </g>
    </svg>
  )
}

function StartPage({ sessions, onNew, onOpen, onDelete, onDuplicate }) {
  const [query, setQuery] = useState("")
  const [sourceFilter, setSourceFilter] = useState("")
  const [modelFilter, setModelFilter] = useState("")
  const loading = sessions === null
  const list = sessions ?? []

  const sourceCounts = useMemo(() => {
    const counts = new Map()
    list.forEach((item) => {
      const source = item.summary?.source ?? "Other"
      counts.set(source, (counts.get(source) ?? 0) + 1)
    })
    return [...counts.entries()]
  }, [list])

  const modelCounts = useMemo(() => {
    const counts = new Map()
    list.forEach((item) => {
      const names = item.summary?.models?.length ? item.summary.models : ["Unknown model"]
      new Set(names).forEach((name) => counts.set(name, (counts.get(name) ?? 0) + 1))
    })
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  }, [list])

  const filtered = useMemo(() => {
    const text = query.trim().toLowerCase()
    return list.filter((item) => {
      if (sourceFilter && (item.summary?.source ?? "Other") !== sourceFilter) return false
      if (modelFilter && !(item.summary?.models ?? []).includes(modelFilter)) return false
      if (!text) return true
      const haystack = [item.name, item.summary?.source, item.summary?.modeShort, ...(item.summary?.models ?? [])]
        .join(" ")
        .toLowerCase()
      return haystack.includes(text)
    })
  }, [list, query, sourceFilter, modelFilter])

  const groups = useMemo(() => {
    const order = ["Today", "Earlier this week", "Earlier"]
    const bucket = new Map(order.map((key) => [key, []]))
    filtered.forEach((item) => bucket.get(sessionDateGroup(item.updatedAt)).push(item))
    return order.map((key) => [key, bucket.get(key)]).filter(([, items]) => items.length)
  }, [filtered])

  const bestR2 = list.length ? Math.max(...list.map((item) => item.summary?.r2 ?? 0)) : null

  return (
    <div className="grid min-h-screen grid-cols-[270px_minmax(0,1fr)] bg-background text-text-primary">
      <aside className="surface-rail sticky top-0 flex h-screen flex-col border-r border-border px-5 py-6">
        <div className="mb-6 flex items-center gap-3">
          <BrandMark className="h-11 w-11 shrink-0 rounded-[12px] shadow-card" />
          <div className="min-w-0">
            <h1 className="truncate text-[15px] font-semibold leading-tight">Hyperelastic</h1>
            <h1 className="truncate text-[15px] font-semibold leading-tight">Calibration</h1>
          </div>
        </div>
        <button
          className="w-full rounded-xl primary-gradient px-4 py-3 text-sm font-bold text-white shadow-primary-glow transition hover:brightness-105"
          onClick={onNew}
        >
          + New Calibration
        </button>

        <div className="mb-2 mt-6 text-[10.5px] font-extrabold uppercase tracking-[0.13em] text-text-disabled">Data Sources</div>
        <button
          className={`flex items-center justify-between rounded-lg px-2.5 py-2 text-[13px] ${!sourceFilter ? "bg-selection-bg font-semibold text-primary" : "text-text-muted hover:bg-subtle"}`}
          onClick={() => setSourceFilter("")}
        >
          <span>All sessions</span>
          <span className="rounded-full border border-border bg-surface px-2 text-[11px] text-text-disabled">{list.length}</span>
        </button>
        {sourceCounts.map(([source, count]) => (
          <button
            key={source}
            className={`flex items-center justify-between rounded-lg px-2.5 py-2 text-[13px] ${sourceFilter === source ? "bg-selection-bg font-semibold text-primary" : "text-text-muted hover:bg-subtle"}`}
            onClick={() => setSourceFilter(sourceFilter === source ? "" : source)}
          >
            <span className="truncate">{source}</span>
            <span className="rounded-full border border-border bg-surface px-2 text-[11px] text-text-disabled">{count}</span>
          </button>
        ))}

        {modelCounts.length > 0 && (
          <>
            <div className="mb-2 mt-5 text-[10.5px] font-extrabold uppercase tracking-[0.13em] text-text-disabled">Models</div>
            {modelCounts.map(([name, count]) => (
              <button
                key={name}
                className={`flex items-center justify-between rounded-lg px-2.5 py-2 text-[13px] ${modelFilter === name ? "bg-selection-bg font-semibold text-primary" : "text-text-muted hover:bg-subtle"}`}
                onClick={() => setModelFilter(modelFilter === name ? "" : name)}
              >
                <span className="truncate">{name}</span>
                <span className="rounded-full border border-border bg-surface px-2 text-[11px] text-text-disabled">{count}</span>
              </button>
            ))}
          </>
        )}

        <div className="flex-1" />
        <div className="border-t border-dashed border-border pt-4 text-[11.5px] leading-relaxed text-text-disabled">
          Sessions capture the full calibration state — dataset selection, model composition, parameters and result curves. Click any row to restore it.
        </div>
      </aside>

      <main className="min-w-0 px-9 py-8">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-[-0.02em]">Calibration Ledger</h1>
            <p className="mt-1 text-[13px] text-text-muted">
              {loading ? "Loading sessions…" : `${list.length} session${list.length === 1 ? "" : "s"}${bestR2 !== null ? ` · best R² ${bestR2.toFixed(4)}` : ""}`}
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="relative">
              <Icon className="absolute left-3 top-1/2 -translate-y-1/2 text-base text-text-disabled">search</Icon>
              <input
                className="h-10 w-64 rounded-xl border border-border-strong bg-surface pl-9 pr-3 text-[13.5px] outline-none focus:border-primary"
                placeholder="Search sessions, models, sources…"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
          </div>
        </div>

        {!loading && list.length === 0 && (
          <div className="grid place-items-center rounded-2xl border-2 border-dashed border-border-strong px-8 py-24 text-center">
            <div>
              <BrandMark className="mx-auto h-16 w-16 rounded-[16px] shadow-card" />
              <h2 className="mt-5 text-[17px] font-semibold">No saved calibrations yet</h2>
              <p className="mx-auto mt-2 max-w-sm text-[13px] leading-relaxed text-text-muted">
                Finish a calibration and it will appear here automatically — dataset selection, model composition, fitted parameters and curves are all saved and can be restored any time.
              </p>
              <button
                className="mt-6 rounded-xl primary-gradient px-6 py-3 text-sm font-bold text-white shadow-primary-glow transition hover:brightness-105"
                onClick={onNew}
              >
                + Start Your First Calibration
              </button>
            </div>
          </div>
        )}

        {!loading && list.length > 0 && filtered.length === 0 && (
          <div className="rounded-2xl border border-border bg-surface px-8 py-16 text-center text-[13px] text-text-muted shadow-card">
            No matching sessions — try a different keyword or clear the filters.
          </div>
        )}

        {groups.map(([groupName, items]) => (
          <div key={groupName}>
            <div className="mb-3 mt-6 flex items-center gap-3 text-[11.5px] font-extrabold uppercase tracking-[0.11em] text-text-disabled">
              {groupName}
              <span className="h-px flex-1 bg-border" />
            </div>
            {items.map((item) => (
              <div
                key={item.id}
                className="mb-2.5 grid cursor-pointer grid-cols-[150px_minmax(0,1fr)_auto_auto] items-center gap-5 rounded-xl border border-border bg-surface p-3 pr-4 shadow-card transition hover:translate-x-0.5 hover:border-primary"
                onClick={() => onOpen(item.id)}
              >
                <div className="relative h-16 overflow-hidden rounded-lg border border-border bg-gradient-to-b from-[#F7FAFF] to-[#F0F5FD]">
                  <SessionSparkline summary={item.summary} />
                </div>
                <div className="min-w-0">
                  <div className="truncate text-[14.5px] font-semibold tracking-[-0.01em]">{item.name}</div>
                  <div className="mt-0.5 truncate text-xs text-text-muted">
                    {item.summary?.source}
                    {item.summary?.modeShort ? ` · ${item.summary.modeShort}` : ""}
                    {item.summary?.points ? ` · ${item.summary.points} pts` : ""}
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {(item.summary?.models ?? []).map((model) => (
                      <span key={model} className="rounded-md border border-[#CBDBFA] bg-selection-bg px-2 py-0.5 text-[10.5px] font-semibold text-primary">
                        {model}
                      </span>
                    ))}
                    {item.summary?.method && (
                      <span className="rounded-md border border-border bg-subtle px-2 py-0.5 text-[10.5px] font-semibold text-text-muted">{item.summary.method}</span>
                    )}
                    {item.summary?.iterations ? (
                      <span className="rounded-md border border-border bg-subtle px-2 py-0.5 text-[10.5px] font-semibold text-text-muted">{item.summary.iterations} iters</span>
                    ) : null}
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-[16px] font-bold ${((item.summary?.r2 ?? 0) >= 0.98) ? "text-success" : "text-[#D97706]"}`}>
                    {(item.summary?.r2 ?? 0).toFixed(4)}
                  </div>
                  <div className="mt-0.5 text-[11.5px] text-text-disabled">{sessionDateLabel(item.updatedAt)}</div>
                </div>
                <div className="flex items-center gap-1.5" onClick={(event) => event.stopPropagation()}>
                  <button
                    className="grid h-8 w-8 place-items-center rounded-lg border border-border-strong bg-surface text-text-muted transition hover:border-primary hover:text-primary"
                    title="Duplicate session"
                    onClick={() => onDuplicate(item.id)}
                  >
                    <Icon className="text-sm">copy</Icon>
                  </button>
                  <button
                    className="grid h-8 w-8 place-items-center rounded-lg border border-border-strong bg-surface text-text-muted transition hover:border-[#DC2626] hover:text-[#DC2626]"
                    title="Delete session"
                    onClick={() => {
                      if (window.confirm(`Delete session "${item.name}"? This cannot be undone.`)) onDelete(item.id)
                    }}
                  >
                    <Icon className="text-sm">delete</Icon>
                  </button>
                  <button
                    className="grid h-8 w-8 place-items-center rounded-lg primary-gradient text-white shadow-primary-glow"
                    title="Open session"
                    onClick={() => onOpen(item.id)}
                  >
                    <Icon className="text-sm">arrow_forward</Icon>
                  </button>
                </div>
              </div>
            ))}
          </div>
        ))}
      </main>
    </div>
  )
}

function Sidebar({ activeStep, onStepChange, onHome }) {
  const items = [
    ["experimental", "Experimental Data", "Select source datasets"],
    ["models", "Model Architecture", "Compose the energy model"],
    ["optimization", "Optimization", "Calibrate parameters"],
    ["prediction", "Prediction", "Reuse and forecast"],
  ]
  const activeIndex = items.findIndex(([key]) => key === activeStep)
  return (
    <aside className="surface-rail sticky top-0 flex h-screen flex-col border-r border-border px-4 py-5">
      <button
        className="group -mx-1 mb-6 flex items-center gap-3 rounded-xl px-2 py-1.5 text-left transition hover:bg-subtle"
        onClick={onHome}
        title="Back to start page"
      >
        <BrandMark className="h-10 w-10 shrink-0 rounded-[11px] shadow-card" />
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-[15px] font-semibold leading-tight">Hyperelastic</h1>
          <h1 className="truncate text-[15px] font-semibold leading-tight">Calibration</h1>
        </div>
        <Icon className="text-base text-text-disabled opacity-0 transition group-hover:opacity-100">home</Icon>
      </button>

      <div className="mb-2 px-1 text-[10px] font-bold uppercase tracking-[0.14em] text-text-disabled">Workflow</div>
      <nav className="relative flex flex-1 flex-col gap-1">
        {items.map(([key, label, description], index) => {
          const active = activeStep === key
          const passed = activeIndex > -1 && index < activeIndex
          const isLast = index === items.length - 1
          return (
            <button
              key={key}
              onClick={() => onStepChange(key)}
              className={`group relative flex items-start gap-3 rounded-xl px-2.5 py-2.5 text-left transition ${
                active ? "bg-selection-bg" : "hover:bg-subtle"
              }`}
            >
              {!isLast && (
                <span
                  className="absolute left-[26px] top-[42px] h-[calc(100%-24px)] w-0.5 rounded-full"
                  style={{ backgroundColor: passed ? "var(--color-primary)" : "var(--color-border-strong)" }}
                />
              )}
              <span
                className={`relative z-10 grid h-7 w-7 shrink-0 place-items-center rounded-full text-[12px] font-bold transition ${
                  active
                    ? "primary-gradient text-white shadow-primary-glow"
                    : passed
                      ? "bg-primary text-white"
                      : "border border-border-strong bg-surface text-text-muted group-hover:border-primary group-hover:text-primary"
                }`}
              >
                {passed ? <Icon className="text-sm">check</Icon> : index + 1}
              </span>
              <span className="min-w-0 flex-1 pt-0.5">
                <span className={`block truncate text-sm font-semibold ${active ? "text-primary" : "text-text-primary"}`}>{label}</span>
                <span className="mt-0.5 block truncate text-[11px] leading-4 text-text-muted">{description}</span>
              </span>
            </button>
          )
        })}
      </nav>

      <div className="mt-3 rounded-xl border border-border bg-surface/80 p-3 shadow-panel">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-text-disabled">Author</div>
            <div className="mt-1 text-sm font-semibold text-text-primary">Chongran Zhao</div>
          </div>
          <span className="rounded-md border border-border bg-subtle px-1.5 py-0.5 text-[10px] font-semibold text-text-muted">Brown</span>
        </div>
        <div className="mt-2 space-y-0.5 text-[11px] leading-4 text-text-muted">
          <div>Ph.D. student in Engineering, Brown University</div>
          <div>M.Eng. in Mechanics, SUSTech</div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-1.5 text-center text-[11px] font-semibold">
          <a className="rounded-lg border border-border bg-surface px-1.5 py-1.5 text-primary transition hover:border-primary hover:bg-selection-bg" href="https://chongran-zhao.github.io" target="_blank" rel="noreferrer">Website</a>
          <a className="rounded-lg border border-border bg-surface px-1.5 py-1.5 text-primary transition hover:border-primary hover:bg-selection-bg" href="https://github.com/Chongran-Zhao" target="_blank" rel="noreferrer">GitHub</a>
          <a className="rounded-lg border border-border bg-surface px-1.5 py-1.5 text-primary transition hover:border-primary hover:bg-selection-bg" href="mailto:chongran_zhao@brown.edu">Email</a>
        </div>
      </div>
    </aside>
  )
}

function Topbar({ activeStep }) {
  const order = ["experimental", "models", "optimization", "prediction"]
  const stepIndex = Math.max(0, order.indexOf(activeStep))
  const title = {
    experimental: "Experimental Data",
    models: "Model Architecture",
    optimization: "Optimization",
    prediction: "Prediction",
  }[activeStep] ?? "Experimental Data"
  const subtitle = {
    experimental: "Select and preview source stress–stretch datasets",
    models: "Compose the strain-energy model and set parameters",
    optimization: "Run the calibration and review convergence",
    prediction: "Reuse fitted parameters and forecast new modes",
  }[activeStep] ?? ""
  return (
    <header className="surface-topbar sticky top-0 z-30 flex h-16 shrink-0 items-center justify-between border-b border-border px-5">
      <div className="flex min-w-0 items-center gap-3.5">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl primary-gradient text-sm font-bold text-white shadow-primary-glow">
          {stepIndex + 1}
        </span>
        <div className="min-w-0">
          <h2 className="truncate text-[17px] font-semibold leading-tight tracking-[-0.01em]">{title}</h2>
          <p className="truncate text-xs text-text-muted">{subtitle}</p>
        </div>
      </div>
      <div className="hidden items-center gap-3 sm:flex">
        <div className="flex items-center gap-1.5">
          {order.map((key, index) => (
            <span
              key={key}
              className="h-1.5 rounded-full transition-all"
              style={{
                width: index === stepIndex ? "22px" : "8px",
                backgroundColor: index <= stepIndex ? "var(--color-primary)" : "var(--color-border-strong)",
              }}
            />
          ))}
        </div>
        <span className="rounded-lg border border-border bg-surface/80 px-2.5 py-1 text-xs font-semibold text-text-muted">
          Step {stepIndex + 1} <span className="text-text-disabled">/ {order.length}</span>
        </span>
      </div>
    </header>
  )
}

function Card({ title, children, className = "" }) {
  return (
    <div className={`min-w-0 rounded-xl border border-border bg-surface p-4 shadow-card ${className}`}>
      {title && (
        <h3 className="mb-3 flex items-center gap-2 text-[15px] font-semibold tracking-[-0.01em]">
          <span className="h-3.5 w-1 rounded-full primary-gradient" />
          {title}
        </h3>
      )}
      {children}
    </div>
  )
}

function Label({ children }) {
  return <div className="mb-1 text-xs font-bold uppercase text-text-muted">{children}</div>
}

function ModeButton({ option, meta, active, onClick }) {
  const label = option.shortLabel ?? option.label
  const disabledForFitting = fittingDisabledFamilies.has(option.family)
  return (
    <button
      className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-left ${
        active
          ? "border-primary bg-selection-bg"
          : disabledForFitting
            ? "cursor-not-allowed border-border bg-subtle opacity-60"
            : "border-border-strong bg-surface hover:bg-subtle"
      }`}
      disabled={disabledForFitting}
      onClick={onClick}
      title={disabledForFitting ? "BT cannot be used for fitting." : undefined}
    >
      <span className={`h-3 w-3 rounded-full ${meta.dot}`} />
      <span className="min-w-0 flex-1">
        <span className={`block truncate text-sm font-semibold ${disabledForFitting ? "text-text-muted" : ""}`}>{formatDisplayLabel(label)}</span>
        <span className="mt-0.5 block truncate text-xs font-normal text-text-muted">
          {disabledForFitting ? "BT cannot be used for fitting" : option.loadingLabel ?? modeFamilyName(option.family)} · {option.points} pts · <StressText display={option.stressDisplay} fallback={option.stressType} />
        </span>
      </span>
      <span className={`grid h-5 w-5 place-items-center rounded border ${active ? "border-primary bg-primary text-white" : disabledForFitting ? "border-border-strong bg-subtle" : "border-border-strong bg-surface"}`}>
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
            <span>{formatBound(param.bounds?.[0], "lower")}</span>
            <span>{formatBound(param.bounds?.[1], "upper")}</span>
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

function StressTypeText({ display, fallback }) {
  if (!display?.symbol) return fallback
  return (
    <>
      <span>{display.label}</span>
      <span className="ml-1 align-middle stress-symbol">
        <LatexInline value={display.symbol} fallback={display.plain ?? fallback} />
      </span>
    </>
  )
}

function stressPlainText(display, fallback) {
  return display?.plain ?? (fallback === "PK1" ? "First Piola-Kirchhoff stress P" : replaceStressWords(fallback))
}

function StressText({ display, fallback }) {
  const text = stressPlainText(display, fallback)
  if (display?.symbol) {
    return (
      <>
        <span>{display.label}</span>
        <span className="ml-1 align-middle stress-symbol">
          <LatexInline value={display.symbol} fallback={text} />
        </span>
      </>
    )
  }
  const rendered = replaceStressWords(text)
  return rendered
}

function replaceStressWords(value) {
  return String(value ?? "")
    .replace(/sigma/g, "σ")
    .replace(/lambda/g, "λ")
    .replace(/gamma/g, "γ")
}

function modeFamilyName(family) {
  return modeOptions.find((item) => item.family === family)?.label ?? family
}

function formatDisplayLabel(label) {
  return String(label ?? "")
    .replace(/\\lambda_2/g, "λ₂")
    .replace(/lambda2/g, "λ₂")
    .replace(/\blambda\b/g, "λ")
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
  const normalized = String(value).replace(/\\boldsymbol\{([^{}]+)\}/g, "\\mathbf{$1}")
  try {
    return katex.renderToString(normalized, {
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

function formatBound(value, side = "upper") {
  if (value === null || value === undefined || value === "") {
    return side === "lower" ? "-Inf" : "Inf"
  }
  const number = Number(value)
  if (number === Infinity) return "Inf"
  if (number === -Infinity) return "-Inf"
  return number.toPrecision(4)
}

function PlotCard({ preview, mode }) {
  const chartGroups = groupPreviewSeries(preview, mode)
  const sourceReference = preview.metadata?.sourceReference ?? preview.metadata?.source ?? "current selection"
  const sourceUrl = preview.metadata?.sourceUrl
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
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <div className="grid gap-3">
          {chartGroups.map((group) => (
            <ScientificChart key={group.key} group={group} />
          ))}
        </div>
      </div>
      <div className="flex items-center justify-end gap-4 border-t border-border px-4 py-3 text-sm">
        <span className="text-text-muted">
          Data sampled from{" "}
          {sourceUrl ? (
            <a className="font-semibold text-primary hover:underline" href={sourceUrl} target="_blank" rel="noreferrer">
              {sourceReference}
            </a>
          ) : (
            <span className="font-semibold text-text-primary">{sourceReference}</span>
          )}
        </span>
        <button className="rounded-lg border border-border-strong bg-surface px-3 py-1.5 font-semibold hover:bg-subtle">Save Plot</button>
      </div>
    </div>
  )
}

function groupPreviewSeries(preview, mode) {
  const series = preview.series?.length
    ? preview.series
    : [{ modeFamily: mode.family, modeLabel: mode.label, modeShortLabel: mode.label, points: preview.points ?? [], axisSymbols: { x: "\\lambda", y: "P_{11}" } }]
  const groups = new Map()
  const xUnit = axisUnitFromLabel(preview.axes?.x, "-")
  const yUnit = axisUnitFromLabel(preview.axes?.y, "MPa")
  series.forEach((item) => {
    const xSymbol = item.axisSymbols?.x ?? "\\lambda"
    const ySymbol = item.axisSymbols?.y ?? "P_{11}"
    const key = `${xSymbol}|${ySymbol}`
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        xSymbol,
        ySymbol,
        xLabel: xSymbol,
        yLabel: ySymbol,
        xUnit,
        yUnit,
        series: [],
      })
    }
    groups.get(key).series.push(item)
  })
  return Array.from(groups.values())
}

function ScientificChart({ group }) {
  const series = group.series
  const hasSecondary = series.some((item) => (item.points ?? []).some((point) => point.x2 !== undefined && point.y2 !== undefined))
  const primarySeries = series.map((item) => ({
    ...item,
    points: (item.points ?? []).map((point) => ({ ...point, plotX: point.x, plotY: point.y })),
  }))
  const secondarySeries = series.map((item) => ({
    ...item,
    points: (item.points ?? [])
      .filter((point) => point.x2 !== undefined && point.y2 !== undefined)
      .map((point) => ({ ...point, plotX: point.x2, plotY: point.y2 })),
  }))
  return (
    <div className={hasSecondary ? "grid min-h-[460px] gap-3 xl:grid-cols-2" : "min-h-[460px]"}>
      <ExperimentalComponentChart series={primarySeries} xLabel={group.xLabel} yLabel={group.yLabel} xUnit={group.xUnit} yUnit={group.yUnit} title={hasSecondary ? "Component P11" : "Selected data"} />
      {hasSecondary && (
        <ExperimentalComponentChart
          series={secondarySeries}
          xLabel={group.xLabel}
          yLabel="P_{22}"
          xUnit={group.xUnit}
          yUnit={group.yUnit}
          title="Component P22"
        />
      )}
    </div>
  )
}

function PreviewTensorOverlay({ series }) {
  const [activeMode, setActiveMode] = useState(series[0]?.mode ?? "")
  useEffect(() => {
    if (!series.some((item) => item.mode === activeMode)) {
      setActiveMode(series[0]?.mode ?? "")
    }
  }, [activeMode, series])
  if (!series.length) return null
  const primary = series.find((item) => item.mode === activeMode) ?? series[0]
  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-border bg-subtle/70 px-3 py-3">
      <div className="grid gap-2">
        <div className="min-w-0">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[10px] font-bold uppercase tracking-wide text-text-muted">Tensor form</div>
            <span className="shrink-0 rounded border border-border bg-white px-1.5 py-0.5 text-[10px] font-semibold text-text-muted">{series.reduce((sum, item) => sum + (item.points?.length ?? 0), 0)} pts</span>
          </div>
          <div className="mt-2 flex max-h-[62px] flex-wrap gap-1 overflow-y-auto pr-0.5">
            {series.map((item, index) => {
              const active = item.mode === primary.mode
              return (
                <button key={previewSeriesKey(item, index)} className={`min-w-0 max-w-full rounded border px-1.5 py-0.5 text-[10px] font-semibold leading-4 ${active ? "border-primary bg-selection-bg text-primary" : "border-border bg-white text-text-muted hover:bg-subtle"}`} onClick={() => setActiveMode(item.mode)}>
                  <span className="mr-1 inline-block h-2 w-2.5 rounded-full align-middle" style={{ backgroundColor: colorForSeries(item.modeFamily, index) }} />
                  <span className="align-middle">{formatDisplayLabel(item.modeShortLabel ?? item.modeLabel)}</span>
                </button>
              )
            })}
          </div>
          <div className="mt-2 truncate text-[10px] text-text-muted">
            {primary.loadingLabel ?? modeFamilyName(primary.modeFamily)} · <LatexInline value={primary.tensorExpressions?.component ?? primary.axisSymbols?.y ?? "P"} fallback={primary.axisSymbols?.y ?? "P"} />
          </div>
        </div>
        <div className="grid min-w-0 grid-cols-2 gap-2">
          <FormulaMini label="F" value={primary.tensorExpressions?.deformationGradient} />
          <FormulaMini label="P" value={primary.tensorExpressions?.firstPkStress} />
        </div>
      </div>
    </div>
  )
}

function FormulaMini({ label, value }) {
  const rendered = useMemo(() => renderFormula(value), [value])
  return (
    <div className="min-h-[42px] min-w-0 rounded border border-border bg-white/80 px-1.5 py-1">
      <div className="mb-1 text-[10px] font-bold uppercase text-text-muted">
        <span className="tensor-letter">{label}</span>
      </div>
      <div className="formula-compact tensor-formula overflow-hidden text-xs">
        {rendered ? <span dangerouslySetInnerHTML={{ __html: rendered }} /> : value ?? "-"}
      </div>
    </div>
  )
}

function formatChartTick(value) {
  if (!Number.isFinite(value)) return ""
  const abs = Math.abs(value)
  if (abs === 0) return "0"
  if (abs >= 100 || abs < 0.01) return value.toExponential(1)
  if (abs >= 10) return value.toFixed(1).replace(/\.0$/, "")
  return value.toFixed(2).replace(/\.?0+$/, "")
}

function plainAxisLabel(value) {
  const subscript = {
    0: "₀",
    1: "₁",
    2: "₂",
    3: "₃",
    4: "₄",
    5: "₅",
    6: "₆",
    7: "₇",
    8: "₈",
    9: "₉",
  }
  return String(value ?? "")
    .replace(/\\lambda/g, "λ")
    .replace(/\\gamma/g, "γ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\mathbf\{([^{}]+)\}/g, "$1")
    .replace(/\\boldsymbol\{([^{}]+)\}/g, "$1")
    .replace(/_\{([^{}]+)\}/g, (_match, digits) => String(digits).replace(/[0-9]/g, (digit) => subscript[digit] ?? digit))
    .replace(/_([0-9]+)/g, (_match, digits) => String(digits).replace(/[0-9]/g, (digit) => subscript[digit] ?? digit))
    .replace(/[{}\\]/g, "")
}

function ExperimentalComponentChart({ series, xLabel, yLabel, xUnit, yUnit, title }) {
  const points = series.flatMap((item) => item.points ?? [])
  const layout = chartLayout(points)
  const { width, height, scaleX, scaleY } = layout
  const legendItems = series.slice(0, 5)

  return (
    <div className="flex h-full min-h-[500px] w-full flex-col overflow-hidden rounded-lg border border-border bg-white">
      <div className="relative min-h-[360px] flex-1">
        <svg className="h-full w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Experimental stress stretch chart">
          <rect x="0" y="0" width={width} height={height} fill="#FFFFFF" />
          <ChartGrid layout={layout} />
          {series.map((item, seriesIndex) => {
            const meta = modeOptions.find((option) => option.family === item.modeFamily) ?? modeOptions[seriesIndex % modeOptions.length]
            const color = colorForSeries(meta.family, seriesIndex)
            const seriesPath = (item.points ?? [])
              .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.plotX)} ${scaleY(point.plotY)}`)
              .join(" ")
            return (
              <g key={previewSeriesKey(item, seriesIndex)}>
                {seriesPath && <path d={seriesPath} fill="none" stroke={color} strokeWidth={chartTheme.fitLineWidth} strokeLinecap="round" strokeLinejoin="round" />}
                {(item.points ?? []).map((point, index) => (
                  <circle key={`${previewSeriesKey(item, seriesIndex)}-${point.plotX}-${point.plotY}-${index}`} cx={scaleX(point.plotX)} cy={scaleY(point.plotY)} r={chartTheme.pointRadius} fill="#FFFFFF" stroke={color} strokeWidth={chartTheme.pointStrokeWidth} />
                ))}
              </g>
            )
          })}
        </svg>
        <ChartAxisLabels xSymbol={xLabel} ySymbol={yLabel} xUnit={xUnit} yUnit={yUnit} />
      </div>
      <div className="border-t border-border bg-white px-4 py-3 text-xs">
        <div className="mb-2 font-semibold text-text-primary">{title}</div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          {legendItems.map((item, index) => (
            <div key={previewSeriesKey(item, index)} className="flex min-w-0 items-center gap-2">
              <span className="h-2.5 w-5 shrink-0 rounded-full" style={{ backgroundColor: colorForSeries(item.modeFamily, index) }} />
              <span className="truncate text-text-muted">{formatDisplayLabel(item.modeShortLabel ?? item.modeLabel ?? "Experimental set")}</span>
            </div>
          ))}
          {series.length > legendItems.length && <div className="text-text-muted">+{series.length - legendItems.length} more</div>}
        </div>
      </div>
    </div>
  )
}

function previewSeriesKey(item, index) {
  return item.mode ?? `${item.modeFamily ?? "series"}-${item.modeLabel ?? item.modeShortLabel ?? "experimental"}-${index}`
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

function BottomBar({ activeStep, rows, selectedBranch, selectedModel, onNext, onStepChange }) {
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
  const order = ["experimental", "models", "optimization", "prediction"]
  const onBack = () => {
    const index = order.indexOf(activeStep)
    if (index > 0) onStepChange(order[index - 1])
  }
  const canGoBack = order.indexOf(activeStep) > 0
  return (
    <footer className="surface-topbar sticky bottom-0 z-20 flex h-16 shrink-0 items-center justify-between border-t border-border px-5">
      <button
        onClick={onBack}
        disabled={!canGoBack}
        className="flex items-center gap-1.5 rounded-xl border border-border-strong bg-surface px-4 py-2 text-sm font-semibold text-text-primary transition hover:bg-subtle disabled:opacity-40"
      >
        <Icon className="text-lg">arrow_back</Icon>
        Back
      </button>
      <div className="flex items-center gap-4">
        <span className="hidden text-sm text-text-muted sm:inline">{status}</span>
        <button
          className="flex items-center gap-1.5 rounded-xl primary-gradient px-5 py-2 text-sm font-semibold text-white shadow-primary-glow transition hover:brightness-105 active:brightness-95"
          onClick={onNext}
        >
          {nextLabel}
          <Icon className="text-lg">arrow_forward</Icon>
        </button>
      </div>
    </footer>
  )
}

createRoot(document.getElementById("root")).render(<App />)
