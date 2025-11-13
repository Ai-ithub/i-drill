
interface DrillingRigIconProps {
  className?: string
  size?: number
}

export default function DrillingRigIcon({ className = '', size = 36 }: DrillingRigIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Ground/Base Platform */}
      <rect x="5" y="88" width="90" height="8" fill="#6B5B4A" rx="1" />
      <rect x="8" y="90" width="84" height="4" fill="#8B7355" />
      
      {/* Main Derrick Structure - Vertical Mast */}
      <rect x="47" y="15" width="6" height="70" fill="#2D3748" />
      <rect x="45" y="15" width="10" height="4" fill="#1A202C" />
      
      {/* Top Crown Block */}
      <rect x="40" y="15" width="20" height="6" fill="#1A202C" />
      <circle cx="50" cy="18" r="2.5" fill="#0F172A" />
      
      {/* Derrick Legs - Triangular Structure */}
      <line x1="15" y1="88" x2="42" y2="25" stroke="#2D3748" strokeWidth="3.5" strokeLinecap="round" />
      <line x1="85" y1="88" x2="58" y2="25" stroke="#2D3748" strokeWidth="3.5" strokeLinecap="round" />
      <line x1="15" y1="88" x2="50" y2="25" stroke="#4A5568" strokeWidth="2" strokeDasharray="3,2" opacity="0.4" />
      <line x1="85" y1="88" x2="50" y2="25" stroke="#4A5568" strokeWidth="2" strokeDasharray="3,2" opacity="0.4" />
      
      {/* Cross Bracing */}
      <line x1="25" y1="55" x2="50" y2="35" stroke="#4A5568" strokeWidth="2" />
      <line x1="75" y1="55" x2="50" y2="35" stroke="#4A5568" strokeWidth="2" />
      <line x1="30" y1="70" x2="50" y2="50" stroke="#718096" strokeWidth="1.5" opacity="0.6" />
      <line x1="70" y1="70" x2="50" y2="50" stroke="#718096" strokeWidth="1.5" opacity="0.6" />
      
      {/* Drill String/Pipe */}
      <line x1="50" y1="21" x2="50" y2="75" stroke="#E2E8F0" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="50" y1="21" x2="50" y2="75" stroke="#CBD5E0" strokeWidth="1.5" strokeDasharray="2,3" opacity="0.5" />
      
      {/* Drill Bit */}
      <circle cx="50" cy="75" r="5" fill="#718096" />
      <circle cx="50" cy="75" r="3" fill="#4A5568" />
      <path d="M 50 70 L 47 75 L 50 80 L 53 75 Z" fill="#2D3748" />
      
      {/* Hook/Traveling Block */}
      <rect x="47" y="21" width="6" height="5" fill="#4A5568" rx="1" />
      <rect x="48" y="22" width="4" height="3" fill="#2D3748" rx="0.5" />
      
      {/* Mud Pumps/Equipment Base */}
      <rect x="12" y="75" width="14" height="13" fill="#2D3748" rx="2" />
      <rect x="14" y="77" width="10" height="9" fill="#1A202C" />
      <rect x="74" y="75" width="14" height="13" fill="#2D3748" rx="2" />
      <rect x="76" y="77" width="10" height="9" fill="#1A202C" />
      
      {/* Equipment Details */}
      <circle cx="19" cy="81" r="2" fill="#4A5568" />
      <circle cx="81" cy="81" r="2" fill="#4A5568" />
    </svg>
  )
}

