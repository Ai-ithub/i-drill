interface LinearGaugeProps {
  label: string
  value: number
  min: number
  max: number
  unit: string
  height?: number
  width?: number
}

export default function LinearGauge({
  label,
  value,
  min,
  max,
  unit,
  height = 200,
  width = 60
}: LinearGaugeProps) {
  // Calculate percentage
  const percentage = ((value - min) / (max - min)) * 100
  const clampedPercentage = Math.max(0, Math.min(100, percentage))

  // Generate tick marks
  const ticks = []
  const numTicks = 5
  for (let i = 0; i <= numTicks; i++) {
    const tickValue = max - ((max - min) * i) / numTicks
    const tickY = (i * height) / numTicks
    ticks.push({ y: tickY, value: Math.round(tickValue) })
  }

  return (
    <div className="flex flex-col items-center">
      {/* Label */}
      <div className="text-white text-xs font-semibold mb-2 text-center">
        {label}
      </div>

      {/* Gauge Container */}
      <div className="relative" style={{ width, height }}>
        {/* Background */}
        <div
          className="absolute inset-0 bg-gray-800 border-2 border-gray-600 rounded"
          style={{ width }}
        />

        {/* Fill */}
        <div
          className="absolute bottom-0 bg-gradient-to-t from-green-500 to-green-300 rounded-b transition-all duration-500"
          style={{
            width,
            height: `${clampedPercentage}%`
          }}
        />

        {/* Tick Marks */}
        {ticks.map((tick, i) => (
          <div
            key={i}
            className="absolute flex items-center"
            style={{ top: tick.y, left: -5, right: -5 }}
          >
            <div className="w-2 h-0.5 bg-white" />
            <div className="flex-1" />
            <span className="text-xs text-gray-400 ml-2">{tick.value}</span>
          </div>
        ))}

        {/* Value Display */}
        <div
          className="absolute left-1/2 transform -translate-x-1/2 bg-black border border-white px-2 py-1 rounded"
          style={{ top: '50%', transform: 'translate(-50%, -50%)' }}
        >
          <div className="text-white text-sm font-bold whitespace-nowrap">
            {Math.round(value)}
          </div>
        </div>
      </div>

      {/* Unit */}
      {unit && (
        <div className="text-gray-400 text-xs mt-1">{unit}</div>
      )}
    </div>
  )
}

