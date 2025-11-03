import { useState, useEffect } from 'react'
import { CircularGauge, LinearGauge } from '@/components/Gauges'

export default function GaugePage() {
  const [sensorData, setSensorData] = useState({
    frequency: 0,
    amplitude: 0,
    psiCompers: 0,
    psiCompersS: 0,
    psiTurbin: 0,
    psiTurbin2: 0,
    relativeTemp: 0,
    surfaceTemp: 0,
    internalTemp: 0,
    pointTemp: 0,
    fluctuatingTemp: 0,
    freezingPoint: 0,
    dewPoint: 0,
    tempVis: 0,
    flashPoint: 0,
    tbn: 0,
    pressureC: 0,
    pressureT: 0
  })

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSensorData({
        frequency: Math.random() * 100,
        amplitude: Math.random() * 1000,
        psiCompers: Math.random() * 1000,
        psiCompersS: Math.random() * 1000,
        psiTurbin: Math.random() * 1000,
        psiTurbin2: Math.random() * 1000,
        relativeTemp: Math.random() * 100,
        surfaceTemp: Math.random() * 100,
        internalTemp: Math.random() * 100,
        pointTemp: Math.random() * 100,
        fluctuatingTemp: Math.random() * 100,
        freezingPoint: Math.random() * 100,
        dewPoint: Math.random() * 100,
        tempVis: Math.random() * 100,
        flashPoint: Math.random() * 100,
        tbn: Math.random() * 100,
        pressureC: Math.random() * 200 - 100,
        pressureT: Math.random() * 200 - 100
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-black text-white p-6">
      {/* Header Dropdowns */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex gap-4">
          <select className="bg-gray-800 border border-gray-600 px-4 py-2 rounded text-sm">
            <option>System</option>
          </select>
          <select className="bg-gray-800 border border-gray-600 px-4 py-2 rounded text-sm">
            <option>Gauge_parameter</option>
          </select>
          <select className="bg-gray-800 border border-gray-600 px-4 py-2 rounded text-sm">
            <option>sensor_parameter</option>
          </select>
        </div>
      </div>

      {/* Frequency Section */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-green-400 italic mb-4">Frequency</h2>
        <div className="grid grid-cols-2 gap-6">
          <CircularGauge
            label="amplitude"
            value={sensorData.amplitude}
            min={0}
            max={100}
            unit=""
          />
          <CircularGauge
            label="frequence"
            value={sensorData.frequency}
            min={0}
            max={1000}
            unit=""
          />
        </div>
      </div>

      {/* Pressure Sections */}
      <div className="grid grid-cols-3 gap-8 mb-8">
        {/* Absolute Pressure */}
        <div>
          <h2 className="text-xl font-bold text-green-400 italic mb-4">Absolute Pressure</h2>
          <div className="space-y-4">
            <CircularGauge
              label="psi-compers"
              value={sensorData.psiCompers}
              min={0}
              max={1000}
              unit=""
            />
            <CircularGauge
              label="psi-turbin"
              value={sensorData.psiTurbin}
              min={0}
              max={1000}
              unit=""
            />
          </div>
        </div>

        {/* Static Pressure */}
        <div>
          <h2 className="text-xl font-bold text-green-400 italic mb-4">Static Pressure</h2>
          <div className="space-y-4">
            <CircularGauge
              label="psi-compers_s"
              value={sensorData.psiCompersS}
              min={0}
              max={1000}
              unit=""
            />
            <CircularGauge
              label="psi-turbin_2"
              value={sensorData.psiTurbin2}
              min={0}
              max={1000}
              unit=""
            />
          </div>
        </div>

        {/* Dynamic Pressure */}
        <div>
          <h2 className="text-xl font-bold text-green-400 italic mb-4">Dynamic Pressure</h2>
          <div className="space-y-4">
            <CircularGauge
              label="psi-comper.s"
              value={sensorData.psiCompers}
              min={0}
              max={1000}
              unit=""
            />
            <CircularGauge
              label="psi-turbin"
              value={sensorData.psiTurbin}
              min={0}
              max={1000}
              unit=""
            />
          </div>
        </div>
      </div>

      {/* Pressure Linear Gauges */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-green-400 italic mb-4">Pressure</h2>
        <div className="flex gap-8 justify-center">
          <LinearGauge
            label="P_C"
            value={sensorData.pressureC}
            min={-100}
            max={100}
            unit=""
          />
          <LinearGauge
            label="P_T"
            value={sensorData.pressureT}
            min={-100}
            max={100}
            unit=""
          />
        </div>
      </div>

      {/* Temperature Section */}
      <div>
        <h2 className="text-xl font-bold text-green-400 italic mb-4">TEMP</h2>
        <div className="grid grid-cols-7 gap-4">
          <LinearGauge
            label="Relative Temp"
            value={sensorData.relativeTemp}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Surface Temp"
            value={sensorData.surfaceTemp}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Internal Temp"
            value={sensorData.internalTemp}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Point Temp"
            value={sensorData.pointTemp}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Fluctuating Temp"
            value={sensorData.fluctuatingTemp}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Freezing Point"
            value={sensorData.freezingPoint}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Dew Point"
            value={sensorData.dewPoint}
            min={0}
            max={100}
            unit=""
          />
        </div>
      </div>

      {/* Viscosity Section */}
      <div className="mt-8">
        <h2 className="text-xl font-bold text-green-400 italic mb-4">Viscosity</h2>
        <div className="grid grid-cols-3 gap-4">
          <LinearGauge
            label="Temp_vis"
            value={sensorData.tempVis}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="Flash Point"
            value={sensorData.flashPoint}
            min={0}
            max={100}
            unit=""
          />
          <LinearGauge
            label="TBN"
            value={sensorData.tbn}
            min={0}
            max={100}
            unit=""
          />
        </div>
      </div>
    </div>
  )
}

