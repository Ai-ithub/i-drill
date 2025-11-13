import PDMTab from '../Dashboard/tabs/PDMTab'

export default function PDM() {
  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">PDM - Predictive Maintenance</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Proactive recommendations for rig maintenance and optimal drilling conditions with automatic execution capability
        </p>
      </div>
      <PDMTab />
    </div>
  )
}

