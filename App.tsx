import React, { useState } from 'react';
import { Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface Transaction {
  amount: number;
  time: number;
  n_transactions_day: number;
  merchant_category: string;
  merchant_name: string;
  customer_id?: string;
  transaction_id: string;
}

interface RiskFactors {
  [key: string]: number;
}

interface PredictionResult {
  transaction_id: string;
  fraud_probability: number;
  is_fraud: boolean;
  risk_factors: RiskFactors;
  timestamp: string;
}

const App: React.FC = () => {
  const [transaction, setTransaction] = useState<Transaction>({
    amount: 0,
    time: 12,
    n_transactions_day: 1,
    merchant_category: 'retail',
    merchant_name: '',
    transaction_id: `TXN${Date.now()}`
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(transaction)
      });

      if (!response.ok) {
        throw new Error('Failed to process transaction');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (probability: number): string => {
    if (probability > 0.7) return 'text-red-600';
    if (probability > 0.3) return 'text-yellow-600';
    return 'text-green-600';
  };

  const pieData = result ? {
    labels: Object.keys(result.risk_factors),
    datasets: [{
      data: Object.values(result.risk_factors),
      backgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56',
        '#4BC0C0',
        '#9966FF'
      ]
    }]
  } : null;

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Financial Fraud Detection System
          </h1>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Transaction Form */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Submit Transaction</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Amount
                </label>
                <input
                  type="number"
                  value={transaction.amount}
                  onChange={e => setTransaction({
                    ...transaction,
                    amount: parseFloat(e.target.value)
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Time (24-hour format)
                </label>
                <input
                  type="number"
                  min="0"
                  max="24"
                  step="0.5"
                  value={transaction.time}
                  onChange={e => setTransaction({
                    ...transaction,
                    time: parseFloat(e.target.value)
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Number of Transactions Today
                </label>
                <input
                  type="number"
                  min="0"
                  value={transaction.n_transactions_day}
                  onChange={e => setTransaction({
                    ...transaction,
                    n_transactions_day: parseInt(e.target.value)
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Merchant Category
                </label>
                <select
                  value={transaction.merchant_category}
                  onChange={e => setTransaction({
                    ...transaction,
                    merchant_category: e.target.value
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  required
                >
                  <option value="retail">Retail</option>
                  <option value="travel">Travel</option>
                  <option value="entertainment">Entertainment</option>
                  <option value="food">Food & Dining</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Merchant Name
                </label>
                <input
                  type="text"
                  value={transaction.merchant_name}
                  onChange={e => setTransaction({
                    ...transaction,
                    merchant_name: e.target.value
                  })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {loading ? 'Processing...' : 'Check for Fraud'}
              </button>
            </form>

            {error && (
              <div className="mt-4 text-red-600">
                {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          {result && (
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900">
                  Fraud Probability
                </h3>
                <div className="mt-2">
                  <div className="relative pt-1">
                    <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
                      <div
                        style={{ width: `${result.fraud_probability * 100}%` }}
                        className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                          result.fraud_probability > 0.7 ? 'bg-red-500' :
                          result.fraud_probability > 0.3 ? 'bg-yellow-500' :
                          'bg-green-500'
                        }`}
                      />
                    </div>
                  </div>
                  <p className={`mt-2 text-lg font-bold ${getRiskColor(result.fraud_probability)}`}>
                    {(result.fraud_probability * 100).toFixed(1)}% Risk
                  </p>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900">
                  Risk Factors
                </h3>
                {pieData && (
                  <div className="mt-4 h-64">
                    <Pie data={pieData} options={{
                      responsive: true,
                      maintainAspectRatio: false
                    }} />
                  </div>
                )}
              </div>

              <div className="text-sm text-gray-500">
                Transaction ID: {result.transaction_id}<br />
                Timestamp: {new Date(result.timestamp).toLocaleString()}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App; 