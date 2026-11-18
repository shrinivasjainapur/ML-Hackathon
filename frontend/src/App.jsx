
import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Crosshair,
  Database,
  Gauge,
  Layers3,
  MapPin,
  Navigation,
  Play,
  RefreshCw,
  Route,
  Satellite,
  Search,
  Zap,
} from 'lucide-react'

import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  useMap,
} from 'react-leaflet'

import 'leaflet/dist/leaflet.css'
import './App.css'

import {
  fetchTrips,
  fetchTripDetails,
  predictRoute,
} from './services/api'


// Convert API coordinates into Leaflet [latitude, longitude] format
function getCoordinates(value) {
  if (!value) return []

  // GeoJSON LineString uses [longitude, latitude]
  if (value.type === 'LineString' && Array.isArray(value.coordinates)) {
    return value.coordinates.map(([lng, lat]) => [lat, lng])
  }

  if (!Array.isArray(value)) return []

  return value.map((point) => {
    if (Array.isArray(point)) {
      return import.meta.env.VITE_COORDINATE_ORDER === 'lnglat'
        ? [Number(point[1]), Number(point[0])]
        : [Number(point[0]), Number(point[1])]
    }

    return [
      Number(point.lat ?? point.latitude),
      Number(point.lng ?? point.lon ?? point.longitude),
    ]
  }).filter(([lat, lng]) =>
    Number.isFinite(lat) && Number.isFinite(lng)
  )
}


// Automatically adjust map to display available points
function FitMap({ points }) {
  const map = useMap()

  useEffect(() => {
    if (points.length === 1) {
      map.setView(points[0], 14)
    } else if (points.length > 1) {
      map.fitBounds(points, { padding: [45, 45] })
    }
  }, [map, points])

  return null
}


// Metric card component
function MetricCard({ icon: Icon, label, value, unit }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">
        <Icon size={18} />
      </div>

      <div>
        <p className="metric-label">{label}</p>

        <div className="metric-value">
          {value ?? '—'}

          {value !== undefined && value !== null && unit && (
            <span>{unit}</span>
          )}
        </div>
      </div>
    </div>
  )
}


function App() {
  // Application state
  const [trips, setTrips] = useState([])
  const [selectedTrip, setSelectedTrip] = useState(null)
  const [result, setResult] = useState(null)

  // GPS trip details
  const [tripDetails, setTripDetails] = useState(null)
  const [loadingTripDetails, setLoadingTripDetails] = useState(false)

  // UI state
  const [search, setSearch] = useState('')
  const [loadingTrips, setLoadingTrips] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [apiStatus, setApiStatus] = useState('connecting')


  // Retrieve GPS observations for a selected trip
  async function loadTripDetails(tripId) {
    if (!tripId) return

    setLoadingTripDetails(true)
    setTripDetails(null)
    setError('')
    setResult(null)

    try {
      const data = await fetchTripDetails(tripId)
      setTripDetails(data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Could not load GPS trip details. Check the backend API.'
      )
    } finally {
      setLoadingTripDetails(false)
    }
  }


  // Load available trips when the application starts
  useEffect(() => {
    let active = true

    async function loadTrips() {
      try {
        const data = await fetchTrips()

        if (!active) return

        setTrips(data)
        setApiStatus('connected')

        const firstTrip = data[0] || null

        setSelectedTrip(firstTrip)

        if (firstTrip) {
          const tripId = firstTrip.id ?? firstTrip.trip_id
          await loadTripDetails(tripId)
        }
      } catch (err) {
        if (!active) return

        setApiStatus('offline')

        setError(
          err.response?.data?.detail ||
          'Could not load trips. Check that the FastAPI server is running and the API URL is correct.'
        )
      } finally {
        if (active) setLoadingTrips(false)
      }
    }

    loadTrips()

    return () => {
      active = false
    }
  }, [])


  // Filter trips using the search box
  const filteredTrips = useMemo(() => {
    return trips.filter((trip) => {
      const id = String(trip.id ?? trip.trip_id ?? '')

      return id.toLowerCase().includes(search.toLowerCase())
    })
  }, [trips, search])


  // GPS coordinates come directly from the trip details API
  const gpsPoints = useMemo(
    () => getCoordinates(tripDetails?.gps_points),
    [tripDetails]
  )


  // Predicted route comes from the prediction API
  const predictedRoute = useMemo(
    () => getCoordinates(result?.predicted_route),
    [result]
  )


  // Optional road network returned by the prediction API
  const roadNetwork = useMemo(
    () => getCoordinates(result?.road_network),
    [result]
  )


  // Combine all available points for map positioning
  const mapPoints = [
    ...gpsPoints,
    ...predictedRoute,
    ...roadNetwork,
  ]

  const mapCenter = mapPoints[0] || [12.9716, 77.5946]


  // Request route prediction
  async function handlePredict() {
    if (!selectedTrip) return

    setPredicting(true)
    setError('')
    setResult(null)

    try {
      const tripId = selectedTrip.id ?? selectedTrip.trip_id

      const data = await predictRoute(tripId)

      setResult(data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Route prediction failed. Check the backend API and try again.'
      )
    } finally {
      setPredicting(false)
    }
  }


  // Prediction metrics
  const metrics = result?.metrics || {}


  return (
    <div className="app-shell">

      {/* SIDEBAR */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-mark">
            <Route size={22} />
          </div>

          <div>
            <h1>PathSnap</h1>
            <span>ROUTE INTELLIGENCE</span>
          </div>
        </div>


        {/* TRIP LIST */}
        <div className="sidebar-section">

          <div className="section-heading">
            <span>DEVELOPMENT DATASET</span>

            <span className="count-badge">
              {trips.length || '—'}
            </span>
          </div>


          <div className="search-box">
            <Search size={16} />

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search trips..."
            />
          </div>


          <div className="trip-list">

            {loadingTrips ? (
              <div className="empty-state">
                Loading trips...
              </div>
            ) : filteredTrips.length === 0 ? (
              <div className="empty-state">
                {trips.length
                  ? 'No matching trips'
                  : 'No trips available'}
              </div>
            ) : (
              filteredTrips.map((trip, index) => {
                const id = trip.id ?? trip.trip_id

                const active = String(
                  selectedTrip?.id ?? selectedTrip?.trip_id
                ) === String(id)

                return (
                  <button
                    className={`trip-item ${active ? 'active' : ''}`}
                    key={id ?? index}
                    onClick={() => {
                      setSelectedTrip(trip)
                      loadTripDetails(id)
                    }}
                  >

                    <span className="trip-number">
                      {String(index + 1).padStart(2, '0')}
                    </span>

                    <span className="trip-info">
                      <strong>Trip {id ?? index + 1}</strong>
                      <small>Student-test trip</small>
                    </span>

                    {active && (
                      <span className="active-dot" />
                    )}

                  </button>
                )
              })
            )}

          </div>
        </div>


        {/* SIDEBAR FOOTER */}
        <div className="sidebar-bottom">

          <div className="privacy-note">
            <Database size={17} />

            <div>
              <strong>Development mode</strong>
              <span>Reference routes are not exposed</span>
            </div>
          </div>

          <div className="sidebar-footer">
            PathSnap · Development Console
          </div>

        </div>
      </aside>


      {/* MAIN CONTENT */}
      <main className="main-content">

        {/* TOP HEADER */}
        <header className="topbar">

          <div>
            <div className="breadcrumb">
              WORKSPACE <span>/</span> DEVELOPMENT
            </div>

            <h2>Route Evaluation</h2>
          </div>

          <div className={`connection-status ${apiStatus}`}>
            <span className="status-dot" />

            {apiStatus === 'connected'
              ? 'API Connected'
              : apiStatus === 'offline'
                ? 'API Offline'
                : 'Connecting'}
          </div>

        </header>


        {/* WELCOME SECTION */}
        <section className="welcome-row">

          <div>
            <p className="eyebrow">
              <Satellite size={14} />
              DISTRIBUTED MAP-MATCHING
            </p>

            <h3>Reconstruct. Evaluate. Understand.</h3>

            <p className="subtitle">
              Select a student-test trip and visualize its GPS observations.
            </p>
          </div>

          <div className="run-indicator">
            <span className="pulse" />
            DEVELOPMENT PREVIEW
          </div>

        </section>


        {/* ERROR MESSAGE */}
        {error && (
          <div className="error-banner">

            <AlertCircle size={18} />

            <span>{error}</span>

            <button
              onClick={() => setError('')}
              aria-label="Dismiss error"
            >
              ×
            </button>

          </div>
        )}


        {/* MAP AND CONTROLS */}
        <section className="workspace-grid">

          {/* MAP PANEL */}
          <div className="map-panel panel">

            <div className="panel-header">

              <div>
                <div className="panel-kicker">
                  SPATIAL VISUALIZATION
                </div>

                <h3>Route Map</h3>
              </div>

              <div className="map-tools">
                <span>
                  <Layers3 size={15} />
                  OpenStreetMap
                </span>
              </div>

            </div>


            {/* LEAFLET MAP */}
            <div className="map-wrap">

              <MapContainer
                center={mapCenter}
                zoom={12}
                scrollWheelZoom
                className="route-map"
              >

                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <FitMap points={mapPoints} />


                {/* ROAD NETWORK */}
                {roadNetwork.length > 1 && (
                  <Polyline
                    positions={roadNetwork}
                    pathOptions={{
                      color: '#64748b',
                      weight: 4,
                      opacity: 0.55,
                    }}
                  />
                )}


                {/* PREDICTED ROUTE */}
                {predictedRoute.length > 1 && (
                  <Polyline
                    positions={predictedRoute}
                    pathOptions={{
                      color: '#8b7cff',
                      weight: 5,
                      opacity: 0.95,
                    }}
                  />
                )}


                {/* ORIGINAL GPS OBSERVATIONS */}
                {gpsPoints.map((point, index) => (
                  <CircleMarker
                    key={`gps-${index}`}
                    center={point}
                    radius={4}
                    pathOptions={{
                      color: '#f5a742',
                      fillColor: '#f5a742',
                      fillOpacity: 0.9,
                      weight: 1,
                    }}
                  />
                ))}

              </MapContainer>


              {/* MAP PLACEHOLDER */}
              {!result && gpsPoints.length === 0 && (
                <div className="map-placeholder">

                  <div className="map-placeholder-icon">
                    <Crosshair size={25} />
                  </div>

                  <strong>
                    {loadingTripDetails
                      ? 'Loading GPS observations'
                      : 'Awaiting GPS data'}
                  </strong>

                  <span>
                    Select a trip to display its GPS observations.
                  </span>

                </div>
              )}

            </div>


            {/* MAP LEGEND */}
            <div className="map-legend">

              <span>
                <i className="legend-dot gps" />
                GPS observations
              </span>

              <span>
                <i className="legend-line predicted" />
                Predicted route
              </span>

              {roadNetwork.length > 1 && (
                <span>
                  <i className="legend-line roads" />
                  Road network
                </span>
              )}

            </div>

          </div>


          {/* CONTROL COLUMN */}
          <aside className="control-column">

            {/* SELECTED TRIP */}
            <div className="panel selected-panel">

              <div className="panel-kicker">
                CURRENT SELECTION
              </div>


              <div className="selected-trip">

                <div className="selected-icon">
                  <MapPin size={20} />
                </div>

                <div>
                  <span>SELECTED TRIP</span>

                  <h3>
                    {selectedTrip
                      ? `Trip ${selectedTrip.id ?? selectedTrip.trip_id}`
                      : 'No trip selected'}
                  </h3>
                </div>

              </div>


              <div className="selection-divider" />


              <div className="selection-detail">
                <span>Dataset</span>
                <strong>Student-test Dataset</strong>
              </div>


              <div className="selection-detail">
                <span>GPS observations</span>

                <strong>
                  {loadingTripDetails
                    ? 'Loading...'
                    : tripDetails?.point_count ?? '—'}
                </strong>
              </div>


              <div className="selection-detail">
                <span>Reference route</span>

                <strong className="hidden-label">
                  Not exposed
                </strong>
              </div>


              {/* PREDICTION BUTTON */}
              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={!selectedTrip || predicting}
              >

                {predicting ? (
                  <>
                    <RefreshCw size={17} className="spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Play size={17} fill="currentColor" />
                    Run Prediction
                    <ArrowRight size={17} />
                  </>
                )}

              </button>

            </div>


            {/* PREDICTION STATUS */}
            <div className="panel output-panel">

              <div className="panel-kicker">
                PREDICTION STATUS
              </div>


              <div className="output-status">

                {result ? (
                  <>
                    <CheckCircle2 size={19} className="success-icon" />
                    <strong>Prediction complete</strong>
                  </>
                ) : predicting ? (
                  <>
                    <RefreshCw size={18} className="spin" />
                    <strong>Processing route...</strong>
                  </>
                ) : (
                  <>
                    <Clock3 size={19} />
                    <strong>Waiting for prediction</strong>
                  </>
                )}

              </div>


              <p>
                {result
                  ? 'The backend returned a prediction for this trip.'
                  : 'GPS observations can be viewed while model integration is in progress.'}
              </p>

            </div>

          </aside>

        </section>


        {/* METRICS SECTION */}
        <section className="metrics-section">

          <div className="metrics-heading">

            <div>
              <div className="panel-kicker">
                MODEL PERFORMANCE
              </div>

              <h3>Evaluation Metrics</h3>
            </div>

            <span className="metrics-note">
              <Activity size={15} />
              Backend response
            </span>

          </div>


          <div className="metrics-grid">

            <MetricCard
              icon={Gauge}
              label="Route Accuracy"
              value={metrics.route_accuracy ?? metrics.accuracy}
              unit={
                metrics.route_accuracy !== undefined ||
                metrics.accuracy !== undefined
                  ? '%'
                  : ''
              }
            />


            <MetricCard
              icon={Navigation}
              label="Mean Error"
              value={metrics.mean_error ?? metrics.mean_distance_error}
              unit={
                metrics.mean_error !== undefined ||
                metrics.mean_distance_error !== undefined
                  ? 'm'
                  : ''
              }
            />


            <MetricCard
              icon={Zap}
              label="Inference Time"
              value={metrics.inference_time ?? metrics.execution_time}
              unit={
                metrics.inference_time !== undefined ||
                metrics.execution_time !== undefined
                  ? 's'
                  : ''
              }
            />


            <MetricCard
              icon={MapPin}
              label="Route Points"
              value={predictedRoute.length || undefined}
              unit={predictedRoute.length ? 'pts' : ''}
            />

          </div>


          {!result && (
            <p className="metrics-empty">
              Metrics will populate after a successful prediction.
              No sample values are displayed.
            </p>
          )}

        </section>


        {/* FOOTER */}
        <footer className="page-footer">

          <span>PATHSNAP DEVELOPMENT</span>

          <span>
            GPS visualization · Ground truth not exposed
          </span>

        </footer>

      </main>

    </div>
  )
}


export default App
