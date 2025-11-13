import RTOTab from '../Dashboard/tabs/RTOTab'

export default function RTO() {
  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">RTO - Real Time Optimization</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Real-time optimization recommendations to improve rate of penetration (ROP), reduce costs, and increase drilling efficiency
        </p>
      </div>
      <RTOTab />
    </div>
  )
}

