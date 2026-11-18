import axios from 'axios'

const api = axios.create({
  // In development Vite proxies /api to FastAPI. Set this to an absolute API
  // URL when the frontend and backend are hosted separately.
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 180_000,
  headers: { Accept: 'application/json' },
})

export async function checkApiReadiness({ signal } = {}) {
  const response = await api.get('/health/ready', { signal, timeout: 5000 })
  return response.data
}

export async function fetchTrips({ signal } = {}) {
  const response = await api.get('/trips', { signal })
  return response.data.trips
}

export async function fetchTripDetails(tripId, { signal } = {}) {
  const response = await api.get(`/trips/${encodeURIComponent(tripId)}`, { signal })
  return response.data
}

export async function predictRoute(tripId, { signal } = {}) {
  const response = await api.post(`/predict/${encodeURIComponent(tripId)}`, null, { signal })
  return response.data
}
