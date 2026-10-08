const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function errorDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .filter((item) => typeof item?.msg === "string")
      .map((item) => item.msg)
      .join("; ");
  }
  return "";
}

async function post(path, payload, failureMessage) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("Unable to reach the API. Check that the backend is running.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`${failureMessage} (HTTP ${response.status}; invalid API response).`);
  }
  if (!response.ok) {
    throw new Error(errorDetail(data?.detail) || `${failureMessage} (HTTP ${response.status}).`);
  }
  return data;
}

export async function runQuery(query) {
  const data = await post("/query", { query }, "Unable to process the query");
  if (!data || typeof data.is_supported !== "boolean" || !Array.isArray(data.result)
      || !Number.isFinite(data.confidence_score) || typeof data.explanation !== "string") {
    throw new Error("The API returned an invalid query response.");
  }
  return data;
}

export async function submitFeedback(payload) {
  return post("/feedback", payload, "Unable to save feedback");
}
