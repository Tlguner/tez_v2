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

import { execSync } from "child_process";

function applyPatchToFile(filePath, patchText) {
  try {
    const patchFile = filePath + ".patch";
    fsSync.writeFileSync(patchFile, patchText, "utf8");

    const out = execSync(`patch "${filePath}" "${patchFile}"`, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (out?.length) console.error(out.toString());

    fsSync.unlinkSync(patchFile);
    console.error("🩹 Patch applied successfully.");
  } catch (err) {
    console.error("❌ Failed to apply patch:", err.message);
  }
}

// --- Gemini bağlantısı (.env içinde GEMINI_API_KEY olmalı) ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

async function generateContentWithRetry(model, prompt, retries = 10, delayMs = 5000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await model.generateContent(prompt);
    } catch (err) {
      const errMsg = err.toString();
      const isRateLimit = errMsg.includes("429") || errMsg.includes("Quota") || errMsg.includes("Too Many Requests");
      const isUnavailable = errMsg.includes("503") || errMsg.includes("UNAVAILABLE") || errMsg.includes("high demand");

      // Detect DAILY quota (RPD) exhaustion — retrying won't help until tomorrow
      const isDailyQuota =
        errMsg.includes("GenerateRequestsPerDayPerProjectPerModel") ||
        errMsg.includes("Per day") ||
        (errMsg.includes("429") && errMsg.includes("limit: 20"));

      if (isDailyQuota) {
        console.error("\n🚫 [GÜNLÜK KOTA DOLDU] Gemini API günlük ücretsiz istek limiti (20 RPD) aşıldı.");
        console.error("⏰ Kota UTC gece yarısında sıfırlanır. Lütfen yarın tekrar deneyin.");
        throw err;
      }

      if ((isRateLimit || isUnavailable) && attempt < retries) {
        let waitTimeMs = delayMs * attempt;
        const match = errMsg.match(/retry in ([\d\.]+)s/i);
        if (match) {
          waitTimeMs = Math.ceil(parseFloat(match[1]) * 1000) + 2000;
        } else if (isUnavailable) {
          waitTimeMs = 15000;
        }
        console.error(`\n⏳ [Server Gemini API] Geçici limit veya yoğunluk hatası alındı (Deneme ${attempt}/${retries}).`);
        console.error(`⚠️ ${Math.round(waitTimeMs / 1000)} saniye beklendikten sonra otomatik olarak tekrar denenecektir...`);
        await new Promise((resolve) => setTimeout(resolve, waitTimeMs));
        continue;
      }
      throw err;
    }
  }
}

function logHealingEvent(event) {
  try {
    const logPath = path.join(__dirname, "healing_log.jsonl");
    const entry = JSON.stringify({
      timestamp: new Date().toISOString(),
      ...event,
    }) + "\n";
    fsSync.appendFileSync(logPath, entry, "utf8");
  } catch (e) {
    console.error("⚠️ Failed to write healing log:", e.message);
  }
}

async function runPythonSelfHealing(pythonCmd, args, scriptPath, mode = "Auto") {
  let attempt = 1;
  const maxAttempts = 3; // Allows running the patched code and trying multiple self-healing loops if needed

  while (attempt <= maxAttempts) {
    try {
      console.error(`▶ [Attempt ${attempt}/${maxAttempts}] Running Python script (${mode})...`);
      const result = await runPython(pythonCmd, args);
      
      // Log successful execution (record if it succeeded directly or after healing)
      logHealingEvent({
        mode,
        script: path.basename(scriptPath),
        attempt,
        strategy: attempt === 1 ? "direct_success" : (attempt === 2 ? "level_1_patch" : "level_2_rewrite"),
        success: true
      });

      return { ok: true, result }; // Başarılıysa anında dön
    } catch (err) {
      const errorText = err.toString();
      console.error(`❌ Python execution failed (Attempt ${attempt}):\n${errorText}`);

      logHealingEvent({
        mode,
        script: path.basename(scriptPath),
        attempt,
        strategy: attempt === 1 ? "level_1_patch" : (attempt === 2 ? "level_2_rewrite" : "failed_max_attempts"),
        success: false,
        error_summary: errorText.substring(0, 200).replace(/\n/g, " ")
      });

      if (attempt === 1) {
        // LEVEL 1 — PATCH MODE
        console.error("🔧 Healing Level 1: Generating small unified patch...");
        const prompt = `Python script crashed with:\n${errorText}\n\nTASK:\n- Fix the script using a SMALL unified diff patch.\n- NO full rewrite.\n- Return ONLY the patch text.\n- NO markdown fences.\n\nScript content:\n${fsSync.readFileSync(scriptPath, "utf8")}`;
        const res = await generateContentWithRetry(model, prompt);
        let patch = res.response?.text()?.trim() || "";
        const fenced = patch.match(/```(?:diff)?\n([\s\S]*?)```/i);
        if (fenced) patch = fenced[1];
        
        if (patch && patch.length > 0) {
          applyPatchToFile(scriptPath, patch);
        } else {
          console.error("⚠️ AI returned empty patch. Moving to next strategy.");
        }
      } 
      else if (attempt === 2) {
        // LEVEL 2 — FULL REWRITE MODE
        console.error("🔥 Healing Level 2: Performing FULL script rewrite...");
        const rewritePrompt = `Rewrite the ENTIRE Python script so that it fixes the error:\n${errorText}\n\nRULES:\n- Output ONLY Python code (NO MARKDOWN).\n- KEEP original behaviour.\n- Keep dataset path unchanged.\n- Ensure JSON printing at end.\n\nOriginal script:\n${fsSync.readFileSync(scriptPath, "utf8")}`;
        const res = await generateContentWithRetry(model, rewritePrompt);
        let newScript = res.response?.text()?.trim() || "";
        const fenced = newScript.match(/```(?:python)?\n([\s\S]*?)```/i);
        if (fenced) newScript = fenced[1];
        
        if (newScript && newScript.length > 0) {
          fsSync.writeFileSync(scriptPath, newScript, "utf8");
          console.error("🆕 Script rewritten successfully.");
        }
      } 
      else {
        // LEVEL 3 — EXIT (Pes Et)
        return { ok: false, error: `Self-healing failed after ${maxAttempts} attempts.` };
      }
      
      attempt++; // Yeni koda geç ve bir sonraki döngüde dene
    }
  }
  return { ok: false, error: "Self-healing failed unexpectedly." };
}

export { runPythonSelfHealing };

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
const PY = process.env.PYTHON_PATH || "python";

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
      process.stderr.write(`\r⏳ Processing: [${bar}] ${percent}%`);
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

    py.on("close", (code) => {
      const START = "<<<JSON_START>>>";
      const END = "<<<JSON_END>>>";

      const startIndex = stdout.indexOf(START);
      const endIndex = stdout.indexOf(END);

      let jsonString = "";
      if (startIndex !== -1 && endIndex !== -1) {
        jsonString = stdout
          .substring(startIndex + START.length, endIndex)
          .trim();
      } else {
        const firstBrace = stdout.indexOf("{");
        const lastBrace = stdout.lastIndexOf("}");
        if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
          jsonString = stdout.substring(firstBrace, lastBrace + 1).trim();
        }
      }

      // If valid JSON was printed to stdout, succeed even if Windows C++ teardown exit code is non-zero
      if (jsonString) {
        try {
          const parsed = JSON.parse(jsonString);
          if (parsed && typeof parsed === "object" && (parsed.ok === true || parsed.accuracy || parsed.metrics || parsed.total_rows)) {
            return resolve(parsed);
          }
        } catch (e) {
          /* fallback to exit code check */
        }
      }

      if (code !== 0) {
        return reject(
          new Error(`Python exited with code ${code}. Stderr:\n${stderr}`)
        );
      }

      if (!jsonString) {
        return reject(
          new Error(
            "JSON markers not found in Python output, and no fallback JSON object found. Ensure Python prints <<<JSON_START>>> and <<<JSON_END>>>."
          )
        );
      }

      try {
        const parsed = JSON.parse(jsonString);
        resolve(parsed);
      } catch (e) {
        reject(
          new Error(
            `Failed to parse JSON output: ${e.message}. Raw output:\n${jsonString}`
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

    const result = await generateContentWithRetry(model, prompt);
    let code = result.response.text();

    const fenced = code.match(/```python\n([\s\S]*?)```/i);
    if (fenced) code = fenced[1];

    const filename = `auto_dl_${Date.now()}.py`;
    const filepath = path.join(__dirname, filename);

    await fs.writeFile(filepath, code, "utf8");
    console.error("📄 AutoDL script yazıldı:", filename);

    let output;

    const heal = await runPythonSelfHealing(PY, [filepath], filepath, "AutoDL");

    if (!heal.ok) {
      console.error("❌ AutoDL Self-Healing failed.");
      return {
        content: [
          {
            type: "text",
            text: `AutoDL failed even after self-healing: ${heal.error}`,
          },
        ],
        structuredContent: { error: heal.error },
      };
    }

    const fullResults = heal.result;
    console.error("🤖 AutoDL successful after self-healing (if needed).");
    // JSON SONUCU KAYDETME BLOĞU

    const resultsDir = path.join(__dirname, "results_dl");
    await fs.mkdir(resultsDir, { recursive: true });

    const outFilename = `autodl_${Date.now()}.json`;
    const outPath = path.join(resultsDir, outFilename);

    try {
      await fs.writeFile(outPath, JSON.stringify(fullResults, null, 2), "utf8");
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
        ...fullResults,
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
      const result = await generateContentWithRetry(model, prompt);
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

        const heal = await runPythonSelfHealing(PY, [scriptPath], scriptPath, "AutoML");        if (!heal.ok) {
          return {
            content: [{ type: "text", text: heal.error }],
            structuredContent: { error: heal.error },
          };
        }

        const rawResult = heal.result;
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

// ==========================================
// 🛠 OTONOM VE MODÜLER PIPELINE TOOL'LARI 🛠
// ==========================================

// 1. Veri seti Keşif Analizi (EDA) Aracı
server.tool(
  "analyze_dataset_tool",
  "Veri setinden 10 satır örnek çeker ve temel istatistikleri döner",
  {
    dataset_path: z.string().optional().default("Phishing_Email.csv"),
  },
  { title: "Dataset Explorer (EDA)" },
  async ({ dataset_path }) => {
    console.error("🔍 analyze_dataset_tool çalıştırılıyor...");
    const rawPath = dataset_path || "Phishing_Email.csv";
    let absPath = path.isAbsolute(rawPath) ? rawPath : path.join(__dirname, rawPath);
    absPath = absPath.replace(/\\/g, "/");

    const edaScript = `
import pandas as pd
import json
import sys

try:
    df = pd.read_csv("${absPath}")
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df.iloc[:, -1].value_counts().to_dict() if len(df.columns) > 0 else {},
        "sample_data": df.head(10).to_dict(orient="records")
    }
    print("<<<JSON_START>>>")
    print(json.dumps(summary, indent=2))
    print("<<<JSON_END>>>")
except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
`;

    const scriptPath = path.join(__dirname, `eda_temp_${Date.now()}.py`);
    await fs.writeFile(scriptPath, edaScript, "utf8");

    try {
      const res = await runPython(PY, [scriptPath]);
      await fs.unlink(scriptPath).catch(() => {});
      return {
        content: [{ type: "text", text: JSON.stringify(res, null, 2) }],
        structuredContent: res,
      };
    } catch (err) {
      await fs.unlink(scriptPath).catch(() => {});
      return {
        content: [{ type: "text", text: `EDA Hatası: ${err.message}` }],
        structuredContent: { error: err.message },
      };
    }
  }
);

// 2. Dinamik Veri Temizleme Aracı
server.tool(
  "clean_dataset_tool",
  "LLM tarafından yazılan veri temizleme kodunu güvenli ve self-healing ile çalıştırır",
  {
    code: z.string(),
  },
  { title: "Dynamic Dataset Cleaner" },
  async ({ code }) => {
    console.error("🧹 clean_dataset_tool çalıştırılıyor...");
    let pythonCode = code.trim();
    const fenced = pythonCode.match(/```(?:python)?\n([\s\S]*?)```/i);
    if (fenced) pythonCode = fenced[1];

    const scriptName = `clean_op_${Date.now()}.py`;
    const scriptPath = path.join(__dirname, scriptName);
    await fs.writeFile(scriptPath, pythonCode, "utf8");

    const heal = await runPythonSelfHealing(PY, [scriptPath], scriptPath, "Dataset Clean");
    await fs.unlink(scriptPath).catch(() => {});

    if (!heal.ok) {
      return {
        content: [{ type: "text", text: `Temizleme Başarısız: ${heal.error}` }],
        structuredContent: { error: heal.error },
      };
    }

    return {
      content: [{ type: "text", text: JSON.stringify(heal.result, null, 2) }],
      structuredContent: heal.result,
    };
  }
);

// 3. Dinamik Özellik Çıkarma Aracı
server.tool(
  "extract_features_tool",
  "LLM tarafından yazılan NLP özellik çıkarma kodunu self-healing ile çalıştırır",
  {
    code: z.string(),
  },
  { title: "Dynamic Feature Extractor" },
  async ({ code }) => {
    console.error("🧬 extract_features_tool çalıştırılıyor...");
    let pythonCode = code.trim();
    const fenced = pythonCode.match(/```(?:python)?\n([\s\S]*?)```/i);
    if (fenced) pythonCode = fenced[1];

    const scriptName = `feature_op_${Date.now()}.py`;
    const scriptPath = path.join(__dirname, scriptName);
    await fs.writeFile(scriptPath, pythonCode, "utf8");

    const heal = await runPythonSelfHealing(PY, [scriptPath], scriptPath, "Feature Extraction");
    await fs.unlink(scriptPath).catch(() => {});

    if (!heal.ok) {
      return {
        content: [{ type: "text", text: `Özellik Çıkarma Başarısız: ${heal.error}` }],
        structuredContent: { error: heal.error },
      };
    }

    return {
      content: [{ type: "text", text: JSON.stringify(heal.result, null, 2) }],
      structuredContent: heal.result,
    };
  }
);

// 4. Dinamik Model Eğitici Aracı
server.tool(
  "train_models_tool",
  "LLM tarafından yazılan ML-Hybrid Ensemble ve AutoML modellerini eğitir",
  {
    code: z.string(),
  },
  { title: "Dynamic Model Trainer" },
  async ({ code }) => {
    console.error("🏋️ train_models_tool çalıştırılıyor...");
    let pythonCode = code.trim();
    const fenced = pythonCode.match(/```(?:python)?\n([\s\S]*?)```/i);
    if (fenced) pythonCode = fenced[1];

    const scriptName = `train_op_${Date.now()}.py`;
    const scriptPath = path.join(__dirname, scriptName);
    await fs.writeFile(scriptPath, pythonCode, "utf8");

    const heal = await runPythonSelfHealing(PY, [scriptPath], scriptPath, "Model Training");
    await fs.unlink(scriptPath).catch(() => {});

    if (!heal.ok) {
      return {
        content: [{ type: "text", text: `Model Eğitimi Başarısız: ${heal.error}` }],
        structuredContent: { error: heal.error },
      };
    }

    return {
      content: [{ type: "text", text: JSON.stringify(heal.result, null, 2) }],
      structuredContent: heal.result,
    };
  }
);

// 5. Dinamik Görselleştirme ve Değerlendirme Aracı
server.tool(
  "evaluate_and_plot_tool",
  "Eğitilen modellerin ROC, Confusion Matrix grafiklerini çizer ve karşılaştırır",
  {
    code: z.string(),
  },
  { title: "Dynamic Model Evaluator & Plotter" },
  async ({ code }) => {
    console.error("📊 evaluate_and_plot_tool çalıştırılıyor...");
    let pythonCode = code.trim();
    const fenced = pythonCode.match(/```(?:python)?\n([\s\S]*?)```/i);
    if (fenced) pythonCode = fenced[1];

    const scriptName = `evaluate_op_${Date.now()}.py`;
    const scriptPath = path.join(__dirname, scriptName);
    await fs.writeFile(scriptPath, pythonCode, "utf8");

    const heal = await runPythonSelfHealing(PY, [scriptPath], scriptPath, "Evaluation & Plots");
    await fs.unlink(scriptPath).catch(() => {});

    if (!heal.ok) {
      return {
        content: [{ type: "text", text: `Görselleştirme Başarısız: ${heal.error}` }],
        structuredContent: { error: heal.error },
      };
    }

    return {
      content: [{ type: "text", text: JSON.stringify(heal.result, null, 2) }],
      structuredContent: heal.result,
    };
  }
);

// --- Transport ile bağlanma (stdio) ---
// Top-level await kullanıyoruz (package.json'da "type":"module" varsa çalışır).
const transport = new StdioServerTransport();
await server.connect(transport);

console.error(
  "🚀 MCP Server connected to STDIO transport. Tools: hello, train_phishing_model, auto_ml_agent, analyze_dataset_tool, clean_dataset_tool, extract_features_tool, train_models_tool, evaluate_and_plot_tool"
);

