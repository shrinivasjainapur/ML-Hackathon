import axios from 'axios'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function fetchTrips() {
  const response = await api.get('/trips')
  const data = response.data
  return Array.isArray(data) ? data : data.trips || []
}

export async function fetchTripDetails(tripId) {
  const response = await api.get(
    `/trips/${encodeURIComponent(tripId)}`
  )
  return response.data
}

export async function predictRoute(tripId) {
  const response = await api.post(
    `/predict/${encodeURIComponent(tripId)}`
  )
  return response.data
}
