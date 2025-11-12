// server.js
import "dotenv/config";
import { spawn } from "child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs/promises";
import fsSync from "fs"; // veya promises ile kullan

import { GoogleGenerativeAI } from "@google/generative-ai";

const server = new McpServer({
  name: "phishing-ml-server",
  version: "1.0.0",
  // ikinci argüman (capabilities) opsiyonel; örnekler bazen bunu ayrı veriyor
});

// __dirname (ESM)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// -- Basit sağlık kontrolü (registerTool ile) --
// hello tool (server.tool biçimi)
server.tool(
  "hello",
  "Health Check",
  { name: z.string().optional() }, // input schema
  {
    title: "Health Check",
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
  async (params) => {
    const output = {
      message: `Hello, ${params?.name ?? "world"}! MCP is alive.`,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(output) }],
      structuredContent: output,
    };
  }
);

// -- Python yolu: .env -> PYTHON_PATH, yoksa Conda fallback --
const PY = process.env.PYTHON_PATH || "D:\\\\Conda\\\\python.exe";
// --- Gemini bağlantısı (.env içinde GEMINI_API_KEY olmalı) ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// -- ML eğitim aracı --
server.tool(
  "train_phishing_model",
  "Train Phishing Model",
  {
    dataset_path: z.string().optional().default("Phishing_Email.csv"),
    model_type: z
      .enum(["logistic", "naive_bayes"])
      .optional()
      .default("logistic"),
    max_features: z.number().optional().default(3000),
    test_size: z.number().optional().default(0.2),
  },
  {
    title: "Train Phishing Model",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  },
  async (params) => {
    const scriptPath = path.join(__dirname, "train_model.py");
    const resolvedDataset = params?.dataset_path ?? "Phishing_Email.csv";
    const dsPath = path.isAbsolute(resolvedDataset)
      ? resolvedDataset
      : path.join(__dirname, resolvedDataset);
    const args = [
      scriptPath,
      dsPath,
      params?.model_type ?? "logistic",
      String(params?.max_features ?? 3000),
      String(params?.test_size ?? 0.2),
    ];

    try {
      await fs.access(dsPath); // throws if not exists
    } catch (e) {
      // return a structured error so client gets something parseable
      return {
        content: [
          { type: "text", text: `ERROR: dataset not found at ${dsPath}` },
        ],
        structuredContent: { error: "dataset_not_found", path: dsPath },
      };
    }

    const res = await runPython(PY, args);

    // 🧠 Log the results in Node console:
    console.log("\n✅ Model training finished successfully.");
    console.log("🎯 Metrics:", JSON.stringify(res.metrics, null, 2));
    console.log("📊 Confusion matrix:", JSON.stringify(res.confusion_matrix));

    // Send structured result to client
    return {
      content: [
        {
          type: "text",
          text: `Training complete.\n${JSON.stringify(res, null, 2)}`,
        },
      ],
      structuredContent: res,
    };
  }
);

function runPython(pythonCmd, args) {
  return new Promise((resolve, reject) => {
    // ensure unbuffered output: insert '-u' if not present
    const argsWithUnbuffer = args.slice(); // copy
    if (!argsWithUnbuffer.includes("-u")) {
      argsWithUnbuffer.unshift("-u"); // python -u script.py ...
    }

    const py = spawn(pythonCmd, argsWithUnbuffer, {
      cwd: __dirname,
      env: process.env,
    });
    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (d) => {
      stdout += d.toString();
      // istersen burada anlık loglayabilirsin:
      // console.log("[PYOUT]", d.toString().trim());
    });

    py.stderr.on("data", (d) => {
      stderr += d.toString();
      console.error("[PY STDERR]", d.toString().trim());
    });

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(`Python exited with code ${code}. Stderr:\n${stderr}`)
        );
      }
      try {
        const parsed = JSON.parse(stdout.trim());
        resolve(parsed);
      } catch (e) {
        reject(
          new Error(
            `Failed to parse JSON from Python stdout: ${e.message}\nStdout:\n${stdout}`
          )
        );
      }
    });

    py.on("error", (e) => {
      reject(new Error("Failed to start Python process: " + e.message));
    });
  });
}

// --- AI'nın kendi ML algoritmasını yazdığı Tool ---
server.tool(
  "auto_ml_agent",
  "AI seçip kendi ML algoritmasını yazsın ve eğitsin",
  {
    dataset_path: z.string().optional().default("Phishing_Email.csv"),
    objective: z
      .string()
      .optional()
      .default("Detect phishing emails with high F1 score"),
  },
  {
    title: "AutoML Generator",
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  },
  async ({ dataset_path, objective }) => {
    try {
      console.log("🤖 AutoML başlatılıyor...");

      // 1️⃣ Prompt - LLM'e gönderilecek açıklama
      // server.js içinde AutoML prompt kısmını bu şekilde düzenle:
      // (server.js içinde auto_ml_agent başlığında prompt'u bu ile değiştir)
      const prompt = `
You are an expert machine learning engineer.

Write a complete, standalone Python script that:
1) MUST load dataset from '${dataset_path}' using pandas.read_csv() with columns:
   - 'Email Text'
   - 'Email Type'
   Labels in file may include variants like 'phishing', 'Phishing Email', 'spam', 'Safe Email', 'ham', 'legitimate', etc.
2) Convert labels robustly to binary:
   map phishing/spam variants => 1, ham/safe/legitimate variants => 0.
   Drop rows with unknown labels or NaNs after mapping.
3) Split into train/test (80/20) using stratify=y.
4) Use TF-IDF with max_features=5000.
5) Try these candidate models and choose the best by cross-validation accuracy (use 5-fold CV):
   - LogisticRegression
   - MultinomialNB
   - RandomForestClassifier
   - LinearSVC
   For each model compute cross_val_score(..., cv=5) and pick the model with highest mean CV accuracy.
6) Retrain the selected (best) model on the full training set and evaluate on the test set.
7) Additionally, compute and output feature importance/explainability:
   - If model is linear (LogisticRegression or LinearSVC), return top 15 TF-IDF terms with highest positive weights for the phishing class.
   - If model is tree-based (RandomForest), return top 15 features by feature_importances_.
   - If model is MultinomialNB, return top 15 log-probability ratios for phishing vs ham.
8) Output ONLY a JSON to stdout (no other prints) with these keys:
   {
     "model_name": "<chosen model name>",
     "cv_accuracy_mean": <float>,
     "cv_accuracy_std": <float>,
     "accuracy": <test accuracy>,
     "precision_phishing": <float>,
     "recall_phishing": <float>,
     "f1_phishing": <float>,
     "top_features": [{"term":"click","weight":0.123}, ... up to 15]
   }
Rules:
- Use only pandas, numpy, sklearn.
- Do NOT simulate or invent data; if file missing raise FileNotFoundError.
- Output JSON via: print(json.dumps(results), flush=True)
- Be robust to label variants and NaN values.
`;

      // 2️⃣ Gemini'den kod üret
      const result = await model.generateContent(prompt);
      const llmCode = result.response.text();

      if (!llmCode || llmCode.trim().length === 0) {
        throw new Error("LLM boş çıktı döndürdü.");
      }

      // 3️⃣ Kod bloğunu ayıkla
      let code = llmCode.trim();
      const fenced = code.match(/```(?:python)?\n([\s\S]*?)```/i);
      if (fenced) code = fenced[1];

      // 4️⃣ Kod dosyasına kaydet
      const scriptName = `auto_model_${Date.now()}.py`;
      const scriptPath = path.join(__dirname, scriptName);
      await fs.writeFile(scriptPath, code, "utf8");
      console.log("🧩 Kod dosyaya yazıldı:", scriptName);

      // 5️⃣ Python ile çalıştır
      try {
        const rawResult = await runPython(PY, [scriptPath]);
        console.log("✅ AutoML tamamlandı.");

        // 💾 Sonuçları kaydet
        const resultsDir = path.join(__dirname, "results");
        await fs.mkdir(resultsDir, { recursive: true });

        const outFilename = `auto_${Date.now()}.json`;
        const outPath = path.join(resultsDir, outFilename);
        try {
          await fs.writeFile(
            outPath,
            JSON.stringify(rawResult, null, 2),
            "utf8"
          );
          console.log("🗂 AutoML result saved to", outPath);
        } catch (e) {
          console.warn("⚠️ Failed to write results file:", e.message);
        }

        // 🔁 Sonuçları MCP'ye gönder
        return {
          content: [
            {
              type: "text",
              text: `AutoML finished. Results saved to ${outPath}`,
            },
          ],
          structuredContent: {
            ...rawResult,
            __result_file: outPath,
          },
        };
      } catch (err) {
        console.error("AutoML tool error:", err);
        return {
          content: [
            { type: "text", text: `AutoML failed: ${String(err.message)}` },
          ],
          structuredContent: {
            error: "runtime_failure",
            message: String(err.message),
            stderr: err.stderr ?? null,
          },
        };
      }
    } catch (err) {
      console.error("AutoML tool error:", err);
      return {
        content: [{ type: "text", text: `AutoML failed: ${err.message}` }],
        structuredContent: { error: err.message },
      };
    }
  }
);

// --- Transport ile bağlanma (stdio) ---
// Top-level await kullanıyoruz (package.json'da "type":"module" varsa çalışır).
const transport = new StdioServerTransport();
await server.connect(transport);

console.error(
  "🚀 MCP Server connected to STDIO transport. Tools: hello, train_phishing_model, auto_ml_agent"
);
