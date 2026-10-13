import { useState } from "react";
import { runQuery, submitFeedback } from "./api";

function Table({ data }) {
  if (!Array.isArray(data) || data.length === 0) return null;
  const columns = Object.keys(data[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>{data.map((row, index) => (
          <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setResponse(null);
    setError("");
    setFeedback("");
    setFeedbackMessage("");
    try {
      const result = await runQuery(query);
      setResponse(result);
    } catch (exc) {
      setResponse(null);
      setError(exc.message);
    } finally {
      setLoading(false);
    }
  }

  async function saveFeedback(e) {
    e.preventDefault();
    if (!response || !feedback.trim()) return;

    setFeedbackSaving(true);
    setFeedbackMessage("");
    setError("");
    try {
      await submitFeedback({
        query: response.query,
        incorrect_interpretation: response.explanation,
        correct_interpretation: feedback.trim(),
        notes: "Frontend feedback submission",
      });
      setFeedback("");
      setFeedbackMessage("Feedback saved successfully.");
    } catch (exc) {
      setFeedbackMessage(exc.message);
    } finally {
      setFeedbackSaving(false);
    }
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">Sales intelligence</p>
        <h1>Intelligent Analytics Query Engine</h1>
        <p>Ask analytical questions about your sales data in natural language.</p>
      </header>
      <section className="panel">
        <form onSubmit={submit}>
          <label htmlFor="query">Natural-language question</label>
          <textarea id="query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Show the top 5 products by revenue" />
          <button disabled={loading || !query.trim()}>{loading ? "Running..." : "Run Query"}</button>
        </form>
        {error && <div className="error" role="alert"><strong>Query request failed.</strong> {error}</div>}
        {response && <div className="result-card">
          <div className="result-header">
            <h2>{response.is_supported === false ? "Unsupported query" : "Result"}</h2>
            <div className="confidence">Confidence: {response.confidence_score.toFixed(3)}</div>
          </div>
          <p className="explanation">{response.explanation}</p>
          {response.generated_logic && <div className="logic"><strong>Generated logic:</strong> {response.generated_logic}</div>}
          {response.is_supported === false
            ? <p className="empty" role="status">This query is unsupported. No calculations were performed.</p>
            : <>
              <Table data={response.result} />
              {response.result.length === 0 && <p className="empty" role="status">No matching records found.</p>}
            </>}
          {response.fallback_used && response.fallback_reason && (
            <p className="provider-note" role="status">
              <strong>Note:</strong> {response.fallback_reason}
            </p>
          )}
          <form className="feedback" onSubmit={saveFeedback}>
            <label htmlFor="feedback">Correct this interpretation (optional)</label>
            <textarea id="feedback" value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="Describe the corrected meaning or field mapping" />
            <button type="submit" disabled={feedbackSaving || !feedback.trim()}>
              {feedbackSaving ? "Saving..." : "Save Feedback"}
            </button>
            {feedbackMessage && <div className="feedback-message" role="status">{feedbackMessage}</div>}
          </form>
        </div>}
      </section>
    </main>
  );
}
