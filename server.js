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

    // 🧠 Log the results in Node console (use stderr to avoid breaking MCP protocol):
    console.error("\n✅ Model training finished successfully.");
    console.error("🎯 Metrics:", JSON.stringify(res.metrics, null, 2));
    console.error("📊 Confusion matrix:", JSON.stringify(res.confusion_matrix));

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
    // Progress bar state
    let progressValue = 0;
    const TOTAL_STEPS = 10; // Her adımı buna bölüyoruz

    function renderProgress() {
      const percent = Math.min(
        100,
        Math.floor((progressValue / TOTAL_STEPS) * 100)
      );
      const barLength = Math.floor(percent / 5);
      const bar = "█".repeat(barLength) + "-".repeat(20 - barLength);
      process.stdout.write(`\r⏳ Processing: [${bar}] ${percent}%`);
    }

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
      const text = d.toString().trim();

      // Progress tetikleyici
      if (text.startsWith("[STEP]")) {
        progressValue += 1;
        renderProgress();
      }

      console.error("\n[PY LOG]", text);
      stdout += text + "\n";
    });

    py.stderr.on("data", (d) => {
      stderr += d.toString();
      console.error("[PY STDERR]", d.toString().trim());
    });

    // In server.js -> runPython -> py.on("close", ...)

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(`Python exited with code ${code}. Stderr:\n${stderr}`)
        );
      }
      try {
        // Find ALL JSON-like sections in stdout — non-greedy
        const matches = stdout.match(/\{[\s\S]*?\}/g);

        if (!matches || matches.length === 0) {
          throw new Error(
            "Python script finished, but no JSON object was found."
          );
        }

        // ALWAYS take the LAST JSON block — Python prints logs first, JSON last
        let jsonString = matches[matches.length - 1];

        // Clean any trailing junk around JSON
        const firstBrace = jsonString.indexOf("{");
        const lastBrace = jsonString.lastIndexOf("}");

        if (firstBrace === -1 || lastBrace === -1) {
          throw new Error("Malformed JSON block detected.");
        }

        jsonString = jsonString.substring(firstBrace, lastBrace + 1);

        // Parse JSON
        const parsed = JSON.parse(jsonString);
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
server.tool(
  "auto_dl_agent",
  "AI otomatik Deep Learning modeli oluştursun ve eğitsin",
  {
    dataset_path: z.string().optional().default("Phishing_Email.csv"),
    objective: z
      .string()
      .optional()
      .default("Train deep learning models for phishing detection"),
  },
  {
    title: "Auto Deep Learning Generator",
    readOnlyHint: false,
  },
  async ({ dataset_path, objective }) => {
    console.error("🤖 AutoDL başlatılıyor...");

    const rawPath = dataset_path || "Phishing_Email.csv";
    let absPath = path.isAbsolute(rawPath)
      ? rawPath
      : path.join(__dirname, rawPath);
    absPath = absPath.replace(/\\/g, "/");

    // Check if exists
    try {
      await fs.access(absPath);
    } catch (e) {
      return {
        content: [{ type: "text", text: `Dataset bulunamadı: ${absPath}` }],
        structuredContent: { error: "dataset_not_found", path: absPath },
      };
    }

    // PROMPT YÜKLEME (AutoDL)
    const autodlPromptTemplate = await fs.readFile(
      path.join(__dirname, "prompts/autodl_prompt.md"),
      "utf8"
    );

    // Placeholder doldur
    const prompt = autodlPromptTemplate.replaceAll("${absPath}", absPath);

    const result = await model.generateContent(prompt);
    let code = result.response.text();

    const fenced = code.match(/```python\n([\s\S]*?)```/i);
    if (fenced) code = fenced[1];

    const filename = `auto_dl_${Date.now()}.py`;
    const filepath = path.join(__dirname, filename);

    await fs.writeFile(filepath, code, "utf8");
    console.error("📄 AutoDL script yazıldı:", filename);

    let output;
    try {
      output = await runPython(PY, [filepath]);
    } catch (err) {
      return {
        content: [{ type: "text", text: `Python error: ${err.message}` }],
        structuredContent: { error: err.message },
      };
    }
    //  JSON SONUCU KAYDETME BLOĞU

    const resultsDir = path.join(__dirname, "results_dl");
    await fs.mkdir(resultsDir, { recursive: true });

    const outFilename = `autodl_${Date.now()}.json`;
    const outPath = path.join(resultsDir, outFilename);

    try {
      await fs.writeFile(outPath, JSON.stringify(output, null, 2), "utf8");
      console.error("📁 AutoDL result saved to:", outPath);
    } catch (e) {
      console.warn("⚠️ Failed to save AutoDL result:", e.message);
    }

    // Response to MCP client
    return {
      content: [
        {
          type: "text",
          text: `AutoDL finished. Results saved to ${outPath}`,
        },
      ],
      structuredContent: {
        ...output,
        __result_file: outPath,
      },
    };
  }
);

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
      console.error("🤖 AutoML başlatılıyor...");
      // --- KRİTİK DÜZELTME BAŞLANGICI ---

      // 1. Gelen parametre boşsa varsayılanı kullan
      const rawPath = dataset_path || "Phishing_Email.csv";

      // 2. Dosya yolunu mutlak (absolute) yola çevir (E:\tez\proje\...)
      let absoluteDatasetPath = path.isAbsolute(rawPath)
        ? rawPath
        : path.join(__dirname, rawPath);

      // 3. Windows ters slash (\) işaretlerini düz slash (/) yap.
      // Python string içinde "\" karakterini kaçış dizisi sanıp hata verebiliyor.
      absoluteDatasetPath = absoluteDatasetPath.replace(/\\/g, "/");

      console.error(`📂 Hedef Veri Seti Yolu: ${absoluteDatasetPath}`);

      // 4. Dosya gerçekten diskte var mı kontrol et?
      try {
        await fs.access(absoluteDatasetPath);
      } catch (e) {
        throw new Error(
          `Veri seti bulunamadı! Lütfen yolu kontrol et: ${absoluteDatasetPath}`
        );
      }
      // --- KRİTİK DÜZELTME BİTİŞİ ---

      // Prompt içinde artık 'absoluteDatasetPath' kullanıyoruz:

      // 1️⃣ Prompt - LLM'e gönderilecek açıklama
      // server.js içinde AutoML prompt kısmını bu şekilde düzenle:
      // (server.js içinde auto_ml_agent başlığında prompt'u bu ile değiştir)
      // PROMPT YÜKLEME (AutoML)
      const automlPromptTemplate = await fs.readFile(
        path.join(__dirname, "prompts/automl_prompt.md"),
        "utf8"
      );

      // Template değişkenlerini doldur
      const prompt = automlPromptTemplate
        .replaceAll("${absoluteDatasetPath}", absoluteDatasetPath)
        .replaceAll("${dataset_path}", dataset_path)
        .replaceAll("${objective}", objective);

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
      console.error("🧩 Kod dosyaya yazıldı:", scriptName);

      // 5️⃣ Python ile çalıştır
      try {
        console.error("Running AutoML script:", scriptPath); // ✔ DOĞRU YER

        const rawResult = await runPython(PY, [scriptPath]);
        console.error("✅ AutoML tamamlandı.");

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
          console.error(" AutoML result saved to", outPath);
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
