import DVRTab from '../Dashboard/tabs/DVRTab'

export default function DVR() {
  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">DVR - Data Validation & Reconciliation</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Validation, reconciliation, and processing of drilling data with anomaly detection and analytical reporting
        </p>
      </div>
      <DVRTab />
    </div>
  )
}

