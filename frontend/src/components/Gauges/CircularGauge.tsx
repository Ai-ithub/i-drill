interface CircularGaugeProps {
  label: string
  value: number
  min: number
  max: number
  unit: string
  size?: number
  color?: string
}

export default function CircularGauge({
  label,
  value,
  min,
  max,
  unit,
  size = 200,
  color
}: CircularGaugeProps) {
  // Calculate angle (gauge goes from -135° to +135° = 270° total)
  const percentage = ((value - min) / (max - min)) * 100
  const angle = -135 + (percentage * 270) / 100

  // Generate tick marks
  const ticks = []
  const numTicks = 9
  for (let i = 0; i <= numTicks; i++) {
    const tickAngle = -135 + (i * 270) / numTicks
    const tickValue = min + ((max - min) * i) / numTicks
    ticks.push({ angle: tickAngle, value: tickValue })
  }

  const strokeColor = color ?? '#4CAF50'

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        {/* Outer Circle */}
        <svg className="w-full h-full" viewBox="0 0 200 200">
          {/* Background Arc */}
          <path
            d="M 30 170 A 85 85 0 1 1 170 170"
            fill="none"
            stroke="#333"
            strokeWidth="20"
          />
          
          {/* Value Arc */}
          <path
            d="M 30 170 A 85 85 0 1 1 170 170"
            fill="none"
            stroke={strokeColor}
            strokeWidth="20"
            strokeDasharray={`${(percentage * 445) / 100} 445`}
            strokeLinecap="round"
          />

          {/* Center Circle */}
          <circle cx="100" cy="100" r="70" fill="#1a1a1a" stroke="#666" strokeWidth="2" />

          {/* Tick Marks */}
          {ticks.map((tick, i) => {
            const radians = (tick.angle * Math.PI) / 180
            const x1 = 100 + 75 * Math.cos(radians)
            const y1 = 100 + 75 * Math.sin(radians)
            const x2 = 100 + 85 * Math.cos(radians)
            const y2 = 100 + 85 * Math.sin(radians)
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#999"
                strokeWidth="1"
              />
            )
          })}

          {/* Needle */}
          <line
            x1="100"
            y1="100"
            x2={100 + 60 * Math.cos((angle * Math.PI) / 180)}
            y2={100 + 60 * Math.sin((angle * Math.PI) / 180)}
            stroke="#fff"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="5" fill="#fff" />

          {/* Value Display */}
          <text
            x="100"
            y="130"
            textAnchor="middle"
            fill="#fff"
            fontSize="24"
            fontWeight="bold"
          >
            {Math.round(value)}
          </text>

          {/* Min/Max Labels */}
          <text x="30" y="185" textAnchor="middle" fill="#999" fontSize="12">
            {min}
          </text>
          <text x="170" y="185" textAnchor="middle" fill="#999" fontSize="12">
            {max}
          </text>
        </svg>
      </div>

      {/* Label */}
      <div className="mt-2 text-center">
        <div className="text-white font-semibold text-sm">{label}</div>
        {unit && <div className="text-gray-400 text-xs">{unit}</div>}
      </div>
    </div>
  )
}

