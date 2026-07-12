// client.js - MCP + Gemini AI Agent Entegrasyonu
import "dotenv/config";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { input, select } from "@inquirer/prompts";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { generateText, jsonSchema } from "ai";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { execSync } from "child_process";

const GLOBAL_TIMEOUT_MS = 48 * 60 * 60 * 1000; // 48 Hours (172800000 ms)

// 1. MCP İstemcisini Hazırla
const mcp = new Client(
  { name: "phishing-agent", version: "1.0.0" },
  { capabilities: {} },
);

// Enforce GLOBAL_TIMEOUT_MS on all requests by default to prevent CLI timeouts during long training
const originalRequest = mcp.request.bind(mcp);
mcp.request = (request, resultSchema, options) => {
  return originalRequest(request, resultSchema, {
    timeout: GLOBAL_TIMEOUT_MS,
    ...options,
  });
};

const transport = new StdioClientTransport({
  // server.js'i child process olarak başlat
  command: "node",
  args: ["server.js"],
  stderr: "inherit",
});

// 2. Gemini Modelini Hazırla (API anahtarı .env'den otomatik alınır)
const google = createGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
});

// Güvenli kapatma fonksiyonu (Önceki kodunuzdan alındı)
async function safeShutdown(code = 0) {
  try {
    await mcp.disconnect();
  } catch (e) {
    /* ignore */
  }
  try {
    if (transport && typeof transport.child?.kill === "function") {
      transport.child.kill();
    }
  } catch (e) {
    /* ignore */
  }
  setTimeout(() => process.exit(code), 500);
}

// --- Fonksiyonlar ---

// 3. LLM'in Aracı (Tool) Kullanma Mantığı
// client.js (handleQuery fonksiyonunun tamamı)

async function handleQuery(tools) {
  const query = await input({
    message: "Agent'a komutunuz (Örn: Logistic modeli 5000 özellik ile eğit)",
  });

  // ✅ ModelMessage formatı
  let currentMessages = [
    { role: "user", content: [{ type: "text", text: query }] },
  ];

  for (let i = 0; i < 5; i++) {
    const messages = [...currentMessages];

    const gen = await generateText({
      model: google("gemini-2.5-flash"),
      messages, // now correctly formatted
      tools: tools.reduce((obj, tool) => {
        obj[tool.name] = {
          description: tool.description,
          inputSchema: jsonSchema(tool.inputSchema),
          execute: async (args) => {
            console.log(
              `\n🤖 Agent, MCP Tool'u çağırıyor: ${
                tool.name
              } (Argümanlar: ${JSON.stringify(args)})`,
            );
            return await mcp.callTool(
              {
                name: tool.name,
                arguments: args,
              },
              undefined,
              { timeout: GLOBAL_TIMEOUT_MS }, // <-- Buraya da eklersek sohbet modu da ölümsüz olur
            );
          },
        };
        return obj;
      }, {}),
    });

    // Debug
    console.log("\n----- LLM raw output (debug) -----");
    console.log("text:", gen.text);
    console.log(
      "toolCalls:",
      JSON.stringify(
        gen.toolCalls ?? gen.toolCallsRaw ?? gen.tool_calls ?? [],
        null,
        2,
      ),
    );
    console.log("----- end debug -----\n");

    const toolCallsRaw =
      gen.toolCalls ??
      gen.toolCallsRaw ??
      gen.tool_calls ??
      gen.toolCalls ??
      [];

    if (toolCallsRaw && toolCallsRaw.length > 0) {
      const toolCall = toolCallsRaw[0];
      const toolName =
        toolCall.toolName ??
        toolCall.name ??
        toolCall.tool ??
        toolCall.tool_name ??
        toolCall.action;

      let argsObj =
        toolCall.args ??
        toolCall.arguments ??
        toolCall.kwargs ??
        toolCall.input ??
        toolCall.inputs ??
        toolCall.parameters ??
        toolCall.params ??
        null;

      if (typeof argsObj === "string") {
        try {
          argsObj = JSON.parse(argsObj);
        } catch {
          const match = argsObj.match(/\{[\s\S]*\}/);
          if (match) {
            try {
              argsObj = JSON.parse(match[0]);
            } catch {
              argsObj = null;
            }
          }
        }
      }

      if (!argsObj || typeof argsObj !== "object") {
        console.warn("Invalid args, using fallback defaults.");
        argsObj = {
          dataset_path: "Phishing_Email.csv",
          model_type: "logistic",
          max_features: 3000,
          test_size: 0.2,
        };
      }

      try {
        const result = await mcp.callTool(
          {
            name: toolName,
            arguments: argsObj,
          },
          undefined,
          { timeout: GLOBAL_TIMEOUT_MS },
        );
        // 🧠 ALWAYS SHOW STRUCTURED RESULTS FROM TOOLS
        if (result?.structuredContent) {
          console.log("\n📊 MCP Tool Result:");
          console.log(JSON.stringify(result.structuredContent, null, 2));
        } else {
          console.log("\n⚠️ No structured result returned:");
          console.log(JSON.stringify(result, null, 2));
        }

        // ✅ ModelMessage formatında ekle
        currentMessages.push({
          role: "assistant",
          content: [
            {
              type: "text",
              text: JSON.stringify(
                result.structuredContent ?? result.content ?? result,
              ),
            },
          ],
        });

        console.log(
          "\n✅ Tool çağrısı başarılı. Agent'a sonuç geri bildirildi.",
        );
      } catch (err) {
        console.error("Tool çağrısı başarısız:", err);
        currentMessages.push({
          role: "assistant",
          content: [{ type: "text", text: `ERROR: ${String(err)}` }],
        });
        return;
      }
    } else if (gen.text) {
      console.log("\n⭐ AGENT'IN NİHAİ YORUMU (Metrik Analizi): ⭐");
      console.log(gen.text);
      return;
    } else {
      console.log("\n[HATA] Agent bir şey üretemedi veya döngü hatası.");
      return;
    }
  }

  console.log("\n[Döngü Bitti] Agent bir sonuca ulaşamadı.");
}

async function main() {
  try {
    await mcp.connect(transport);

    console.log("✅ Client, MCP server'a bağlandı.");

    // GLOBAL timeout is applied automatically via the overridden request method

    // Agent'ın kullanabileceği araçları çek
    const [{ tools }] = await Promise.all([mcp.listTools()]);
    console.log(
      "🤖 Agent'ın Kullanılabilir Tool'ları:",
      tools.map((t) => t.name),
    );

    console.log("\n--- Gemini AI Agent Başlatıldı ---");
    while (true) {
      const option = await select({
        message: "Yapmak istediğiniz işlem",
        choices: [
          { name: "Agent'a Komut Ver (Query)", value: "Query" },
          { name: "AutoML: AI kendi ML algoritmasını yazsın", value: "AutoML" },
          {
            name: "AutoDL: AI kendi Deep Learning kodunu yazsın",
            value: "AutoDL",
          },

          { name: "Train Model (Manuel)", value: "TrainModel" },
          { name: "Health Check (Hello Tool)", value: "Hello" },
          { name: "Çıkış", value: "Exit" },
        ],
      });

      switch (option) {
        case "Query":
          await handleQuery(tools);
          break;
        case "TrainModel": {
          const dataset_path = await input({
            message: "Dataset yolu:",
            default: "Phishing_Email.csv",
          });
          const model_type = await select({
            message: "Model Tipi:",
            choices: [
              { name: "Logistic Regression", value: "logistic" },
              { name: "Naive Bayes", value: "naive_bayes" },
            ],
            default: "logistic",
          });
          const max_features = Number(
            await input({
              message: "Max Features:",
              default: "3000",
            }),
          );
          const test_size = Number(
            await input({
              message: "Test Size (0.1 - 0.9):",
              default: "0.2",
            }),
          );

          console.log("\n🚀 Model eğitimi başlatılıyor...");
          try {
            const result = await mcp.callTool(
              {
                name: "train_phishing_model",
                arguments: {
                  dataset_path,
                  model_type,
                  max_features,
                  test_size,
                },
              },
              undefined,
              { timeout: GLOBAL_TIMEOUT_MS },
            );

            console.log("\n📊 Eğitim Sonucu:");
            console.log(
              JSON.stringify(
                result.structuredContent ?? result.content ?? result,
                null,
                2,
              ),
            );
          } catch (err) {
            console.error("❌ Eğitim hatası:", err);
          }
          break;
        }
        case "AutoDL": {
          const dataset_path = await input({
            message: "Dataset yolu (varsayılan Enter)",
            initial: "Phishing_Email.csv",
          });

          console.log(
            "\n🚀 [Modüler Pipeline] Otonom Akış Başlatılıyor (DL Modu)...",
          );

          try {
            await runModularPipeline(dataset_path, "DL");
          } catch (err) {
            console.error("❌ Modüler pipeline başarısız:", err);
          }

          break;
        }

        case "Hello": {
          try {
            console.log("⏳ Health Check çalıştırılıyor...");
            const result = await mcp.callTool({
              name: "hello",
              arguments: {},
            });
            console.log("\n✅ Tool Sonucu:");
            console.log(
              JSON.stringify(
                result.structuredContent ?? result.content ?? result,
                null,
                2,
              ),
            );
          } catch (err) {
            console.error("❌ Tool hatası:", err);
          }
          break;
        }
        // main() içindeki switch'e ekle:
        case "AutoML": {
          const dataset_path = await input({
            message: "Dataset yolu (enter ile varsayılan kullanılır)",
            initial: "Phishing_Email.csv",
          });

          console.log(
            "\n🚀 [Modüler Pipeline] Otonom Akış Başlatılıyor (ML Modu)...",
          );

          try {
            await runModularPipeline(dataset_path);
          } catch (err) {
            console.error("❌ Modüler pipeline başarısız:", err);
          }
          break;
        }

        case "Exit":
          await safeShutdown(0);
          return;
      }
    }
  } catch (e) {
    console.error("Client Error:", e);
    await safeShutdown(1);
  }
}

// ==========================================
// 🛡️ OTONOM MODÜLER PIPELINE ANA AKIŞI 🛡️
// ==========================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Gemini API Kota Koruyucu ve Akıllı Yeniden Deneme (Smart Retry & Backoff)
async function generateTextWithRetry(options, retries = 10, delayMs = 5000) {
  // maxRetries: 0 must come AFTER the spread so it's never overridden by options.
  const mergedOptions = { ...options, maxRetries: 0 };

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await generateText(mergedOptions);
    } catch (err) {
      const errMsg = err.toString();
      const isRateLimit =
        errMsg.includes("429") ||
        errMsg.includes("Quota") ||
        errMsg.includes("Too Many Requests");
      const isUnavailable =
        errMsg.includes("503") ||
        errMsg.includes("UNAVAILABLE") ||
        errMsg.includes("high demand");

      // Detect DAILY quota (RPD) exhaustion
      const isDailyQuota =
        errMsg.includes("GenerateRequestsPerDayPerProjectPerModel") ||
        errMsg.includes("Per day") ||
        (errMsg.includes("429") && errMsg.includes("limit: 20"));

      if (isDailyQuota) {
        console.error(
          "\n\uD83D\uDEAB [GUNLUK KOTA DOLDU] Gemini API gunluk ucretsiz istek limiti (20 RPD) asildi.",
        );
        console.error(
          "\u23F0 Kota UTC gece yarisinda sifirlanir. Lutfen yarin tekrar deneyin.",
        );
        console.error(
          "\uD83D\uDCA1 Ipucu: Program durdu ama state kaydedildi -- kaldigı yerden devam edebilirsiniz.",
        );
        await safeShutdown(1);
        throw err;
      }

      // 503: Global sunucu yuku -- retry hicbir sey degistirmez, sadece RPD kotasi yakar.
      // Ilk 503'te aninda dur. State kaydedildi, kullanici birka dakika sonra tekrar baslatiyor.
      if (isUnavailable) {
        console.error(
          "\n[503 MODEL MESGUL] Gemini sunuculari su an cok yogun.",
        );
        console.error(
          "Retry yapilmiyor -- her deneme gunluk RPD kotasindan dusuyor.",
        );
        console.error(
          "Birkas dakika bekleyip 'node client.js' ile tekrar calistirin.",
        );
        console.error("State kaydedildi, kaldigı yerden devam edecek.");
        await safeShutdown(1);
        throw err;
      }

      // 429 RPM: full retry budget
      if (isRateLimit && attempt < retries) {
        let waitTimeMs = delayMs * attempt;
        const match = errMsg.match(/retry in ([\d\.]+)s/i);
        if (match) {
          waitTimeMs = Math.ceil(parseFloat(match[1]) * 1000) + 2000;
        }
        console.error(
          `\n[429 Rate Limit] Istek limiti asildi (Deneme ${attempt}/${retries}).`,
        );
        console.error(
          `${Math.round(waitTimeMs / 1000)} saniye beklendikten sonra tekrar denenecektir...`,
        );
        await new Promise((resolve) => setTimeout(resolve, waitTimeMs));
        continue;
      }

      console.error("\n[API HATASI] Tum denemeler basarisiz oldu.");
      await safeShutdown(1);
      throw err;
    }
  }
}

// 💾 Durum Kaydedici ve Yükleyici (State-Saver & Resume)
async function saveState(runDir, state) {
  try {
    const statePath = path.join(runDir, "run_state.json");
    await fs.writeFile(statePath, JSON.stringify(state, null, 2), "utf8");
  } catch (e) {
    console.warn("⚠️ Durum kaydedilemedi:", e.message);
  }
}

async function loadState(runDir) {
  try {
    const statePath = path.join(runDir, "run_state.json");
    const content = await fs.readFile(statePath, "utf8");
    return JSON.parse(content);
  } catch {
    return null;
  }
}

// Yarım kalan çalışmayı algıla
async function checkAndGetResumePath(pipelineMode) {
  try {
    const baseDir = path.join(
      __dirname,
      "thesis_pipeline",
      pipelineMode,
      "runs",
    );
    const runs = await fs.readdir(baseDir);

    const runDirs = runs.filter((r) => r.startsWith("run_"));
    if (runDirs.length === 0) return null;

    // Klasörleri son güncellenme/değiştirilme zamanına göre sıralayalım (en güncel en sonda olacak şekilde)
    const runsWithTime = [];
    for (const name of runDirs) {
      try {
        const stats = await fs.stat(path.join(baseDir, name));
        runsWithTime.push({ name, mtime: stats.mtimeMs });
      } catch (err) {
        /* ignore */
      }
    }

    runsWithTime.sort((a, b) => a.mtime - b.mtime);
    if (runsWithTime.length === 0) return null;

    const latestRun = runsWithTime[runsWithTime.length - 1].name;
    const latestRunDir = path.join(baseDir, latestRun);

    const state = await loadState(latestRunDir);
    if (state && state.step_5_viz !== "completed") {
      return { runDir: latestRunDir, state, runName: latestRun };
    }
  } catch (e) {
    /* ignore */
  }
  return null;
}

async function runModularPipeline(datasetPath, pipelineMode = "ML") {
  const dsPath = datasetPath || "Phishing_Email.csv";

  // Yarıda kalmış çalışma kontrolü
  const resumeInfo = await checkAndGetResumePath(pipelineMode);
  let runDir;
  let state = {
    step_0_eda: "pending",
    step_1_clean: "pending",
    step_2_feature: "pending",
    step_3_tune: "pending",
    step_3_train: "pending",
    step_4_eval: "pending",
    step_5_viz: "pending",
  };

  if (resumeInfo) {
    const answer = await select({
      message: `🚨 Yarıda kalmış bir çalışma bulundu (${resumeInfo.runName}). Kaldığı yerden devam etmek ister misiniz?`,
      choices: [
        {
          name: "Evet, kaldığı yerden devam et (Kotaları Korur)",
          value: "yes",
        },
        { name: "Hayır, yeni bir çalışma başlat", value: "no" },
      ],
    });

    if (answer === "yes") {
      runDir = resumeInfo.runDir;
      state = resumeInfo.state;
      console.log(
        `\n🔄 Kaldığı yerden devam ediliyor. Çalışma Klasörü: ${runDir}`,
      );
    }
  }

  const baseDir = path.join(__dirname, "thesis_pipeline", pipelineMode);
  const latestDir = path.join(baseDir, "latest");
  await fs.mkdir(latestDir, { recursive: true });

  if (!runDir) {
    // Yeni çalışma başlatılıyor
    const now = new Date();
    const timestamp =
      now.getFullYear() +
      "-" +
      String(now.getMonth() + 1).padStart(2, "0") +
      "-" +
      String(now.getDate()).padStart(2, "0") +
      "_" +
      String(now.getHours()).padStart(2, "0") +
      "-" +
      String(now.getMinutes()).padStart(2, "0") +
      "-" +
      String(now.getSeconds()).padStart(2, "0");

    const runName = `run_${timestamp}`;
    runDir = path.join(baseDir, "runs", runName);
    await fs.mkdir(runDir, { recursive: true });
    await saveState(runDir, state);
  }

  const modelsDir = path.join(runDir, "models").replace(/\\/g, "/");

  // Initialize start time if not already set (preserves start time on resume)
  state.pipeline_start_time = state.pipeline_start_time || Date.now();
  await saveState(runDir, state);

  // 1️⃣ Adım 0: EDA Analizi
  let edaData;
  if (state.step_0_eda === "completed") {
    console.log(
      "⏭️ Adım 0: EDA Analizi zaten tamamlanmış, diskten yükleniyor...",
    );
    edaData = state.edaData;
  } else {
    console.log("\n🔍 Adım 0: Veri seti keşif analizi (EDA) yapılıyor...");
    const edaRes = await mcp.callTool(
      {
        name: "analyze_dataset_tool",
        arguments: { dataset_path: dsPath },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    edaData = edaRes.structuredContent;
    if (!edaData || edaData.error) {
      throw new Error(
        `EDA adımı başarısız: ${edaData?.error || "Veri çekilemedi"}`,
      );
    }
    state.step_0_eda = "completed";
    state.edaData = edaData;
    await saveState(runDir, state);
    console.log(`📊 EDA Başarılı! Toplam Satır Sayısı: ${edaData.total_rows}`);
  }

  // 2️⃣ Adım 1: Veri Temizleme (Dynamic Clean)
  let cleanResult;
  if (state.step_1_clean === "completed") {
    console.log(
      "⏭️ Adım 1: Veri Temizleme zaten tamamlanmış, diskten yükleniyor...",
    );
    cleanResult = state.cleanResult;
  } else {
    console.log(
      "\n🧹 Adım 1: Veri temizleme Python kodu otonom olarak yazılıyor...",
    );
    const cleanPromptTemplate = await fs.readFile(
      path.join(__dirname, "prompts/clean_prompt.md"),
      "utf8",
    );

    const targetCleanedCsv = path
      .join(runDir, "cleaned_dataset.csv")
      .replace(/\\/g, "/");

    const cleanPrompt = cleanPromptTemplate
      .replaceAll("${dataset_path}", dsPath)
      .replaceAll("${total_rows}", String(edaData.total_rows))
      .replaceAll("${cleaned_dataset_path}", targetCleanedCsv)
      .replaceAll("${eda_summary}", JSON.stringify(edaData, null, 2));

    const cleanGen = await generateTextWithRetry({
      model: google("gemini-2.5-flash"),
      prompt: cleanPrompt,
    });

    console.log("⚡ LLM otonom temizlik kodu üretti. Çalıştırılıyor...");
    const cleanOpRes = await mcp.callTool(
      {
        name: "clean_dataset_tool",
        arguments: { code: cleanGen.text },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    cleanResult = cleanOpRes.structuredContent;
    if (!cleanResult || cleanResult.error) {
      throw new Error(
        `Veri temizleme başarısız: ${cleanResult?.error || "Bilinmeyen Hata"}`,
      );
    }
    state.step_1_clean = "completed";
    state.cleanResult = cleanResult;
    await saveState(runDir, state);
    console.log(
      "✅ Temizlenmiş veri kaydedildi:",
      cleanResult.cleaned_dataset_path,
    );
  }

  // 3️⃣ Adım 2: Özellik Çıkarma (Feature Extraction)
  let featureResult;
  if (state.step_2_feature === "completed") {
    console.log(
      "⏭️ Adım 2: Özellik Çıkarma zaten tamamlanmış, diskten yükleniyor...",
    );
    featureResult = state.featureResult;
  } else {
    console.log("\n🧬 Adım 2: Özellik çıkarma otonom Python kodu yazılıyor...");
    const featurePromptTemplate = await fs.readFile(
      path.join(__dirname, "prompts/feature_prompt.md"),
      "utf8",
    );
    const cleanedRows = cleanResult.cleaned_shape
      ? cleanResult.cleaned_shape[0]
      : edaData.total_rows;

    const isDL = pipelineMode === "DL";
    const targetFeaturesNpy = path
      .join(runDir, isDL ? "X_train.npy" : "X_train.npz")
      .replace(/\\/g, "/");
    const targetLabelsNpy = path
      .join(runDir, "y_train.npy")
      .replace(/\\/g, "/");
    const targetXTest = path
      .join(runDir, isDL ? "X_test.npy" : "X_test.npz")
      .replace(/\\/g, "/");
    const targetYTest = path.join(runDir, "y_test.npy").replace(/\\/g, "/");
    const targetVectorizerPkl = path
      .join(runDir, "vectorizer.pkl")
      .replace(/\\/g, "/");

    const featurePrompt = featurePromptTemplate
      .replaceAll("${cleaned_dataset_path}", cleanResult.cleaned_dataset_path)
      .replaceAll("${cleaned_rows}", String(cleanedRows))
      .replaceAll("${features_npy_path}", targetFeaturesNpy)
      .replaceAll("${labels_npy_path}", targetLabelsNpy)
      .replaceAll("${test_features_path}", targetXTest)
      .replaceAll("${test_labels_path}", targetYTest)
      .replaceAll("${vectorizer_pkl_path}", targetVectorizerPkl)
      .replaceAll(
        "${class_balance}",
        JSON.stringify(cleanResult.class_balance || {}),
      )
      .replaceAll("${mode}", pipelineMode);

    const featureGen = await generateTextWithRetry({
      model: google("gemini-2.5-flash"),
      prompt: featurePrompt,
    });

    console.log("⚡ LLM otonom özellik çıkarma kodu üretti. Çalıştırılıyor...");
    const featureOpRes = await mcp.callTool(
      {
        name: "extract_features_tool",
        arguments: { code: featureGen.text },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    featureResult = featureOpRes.structuredContent;
    if (!featureResult || featureResult.error || featureResult.ok !== true) {
      throw new Error(
        `Özellik çıkarma başarısız: ${featureResult?.error || "Bilinmeyen Hata"}`,
      );
    }
    state.step_2_feature = "completed";
    state.featureResult = featureResult;
    await saveState(runDir, state);
    console.log("✅ Özellikler kaydedildi. Model eğitimine hazır.");
  }

  // 4️⃣ Adım 3.1: Model Hiperparametre Optimizasyonu (Tuning)
  const bestParamsJsonPath = path
    .join(runDir, "best_params.json")
    .replace(/\\/g, "/");

  state.step_3_tune =
    state.step_3_tune ||
    (state.step_3_train === "completed" ? "completed" : "pending");

  if (state.step_3_tune === "completed") {
    console.log(
      "⏭️ Adım 3.1: Hiperparametre optimizasyonu zaten tamamlanmış, diskten yükleniyor...",
    );
  } else {
    console.log(
      "\n🏋️ Adım 3.1: Modellerin hiperparametre optimizasyonu (Optuna) otonom kodu yazılıyor...",
    );

    const tunePromptTemplate = await fs.readFile(
      path.join(
        __dirname,
        pipelineMode === "ML"
          ? "prompts/tune_prompt.md"
          : "prompts/tune_dl_prompt.md",
      ),
      "utf8",
    );

    const tunePrompt = tunePromptTemplate
      .replaceAll("${features_npy_path}", featureResult.features_npy_path)
      .replaceAll("${labels_npy_path}", featureResult.labels_npy_path)
      .replaceAll("${best_params_json_path}", bestParamsJsonPath);

    const tuneGen = await generateTextWithRetry({
      model: google("gemini-2.5-flash"),
      prompt: tunePrompt,
    });

    console.log(
      "⚡ LLM otonom tuning kodu üretti. Optuna optimizasyonu çalıştırılıyor...",
    );
    const tuneOpRes = await mcp.callTool(
      {
        name: "train_models_tool",
        arguments: { code: tuneGen.text },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    const tuneResult = tuneOpRes.structuredContent;
    if (!tuneResult || tuneResult.error || tuneResult.ok !== true) {
      throw new Error(
        `Hiperparametre optimizasyonu başarısız: ${tuneResult?.error || "Bilinmeyen Hata"}`,
      );
    }
    state.step_3_tune = "completed";
    state.optuna_statistics = tuneResult.optuna_statistics;
    await saveState(runDir, state);
    console.log(
      "✅ En iyi parametreler bulundu ve kaydedildi:",
      bestParamsJsonPath,
    );
  }

  // 4️⃣ Adım 3.2: Model Eğitimi (Hybrid Ensemble & Auto-Model)
  let trainResult;
  if (state.step_3_train === "completed") {
    console.log(
      "⏭️ Adım 3.2: Model Eğitimi zaten tamamlanmış, diskten yükleniyor...",
    );
    trainResult = state.trainResult;
  } else {
    console.log(
      "\n🏋️ Adım 3.2: ML Hybrid Ensemble ve Auto-Model eğitim kodları otonom yazılıyor...",
    );

    const trainPromptTemplate = await fs.readFile(
      path.join(
        __dirname,
        pipelineMode === "ML"
          ? "prompts/train_prompt.md"
          : "prompts/train_dl_prompt.md",
      ),
      "utf8",
    );

    const modelSavePath = path
      .join(runDir, "best_model.pkl")
      .replace(/\\/g, "/");

    const trainPrompt = trainPromptTemplate
      .replaceAll("${features_npy_path}", featureResult.features_npy_path)
      .replaceAll("${labels_npy_path}", featureResult.labels_npy_path)
      .replaceAll("${vectorizer_pkl_path}", featureResult.vectorizer_pkl_path)
      .replaceAll("${test_features_path}", featureResult.test_features_path)
      .replaceAll("${test_labels_path}", featureResult.test_labels_path)
      .replaceAll("${model_save_path}", modelSavePath)
      .replaceAll("${best_params_json_path}", bestParamsJsonPath)
      .replaceAll("${models_dir}", modelsDir);

    const trainGen = await generateTextWithRetry({
      model: google("gemini-2.5-flash"),
      prompt: trainPrompt,
    });

    console.log("⚡ LLM otonom eğitim kodu üretti. Modeller eğitiliyor...");
    const trainOpRes = await mcp.callTool(
      {
        name: "train_models_tool",
        arguments: { code: trainGen.text },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    trainResult = trainOpRes.structuredContent;
    if (!trainResult || trainResult.error) {
      throw new Error(
        `Model eğitimi başarısız: ${trainResult?.error || "Bilinmeyen Hata"}`,
      );
    }
    state.step_3_train = "completed";
    state.trainResult = trainResult;
    await saveState(runDir, state);
    console.log("✅ Modeller başarıyla eğitildi ve kaydedildi.");
  }

  // 5️⃣ Adım 4: Değerlendirme & Grafikler (ROC, Confusion Matrix)
  console.log(
    "\n📊 Adım 4: Değerlendirme ve grafik oluşturma otonom kodu yazılıyor...",
  );
  const evalPromptTemplate = await fs.readFile(
    path.join(__dirname, "prompts/eval_prompt.md"),
    "utf8",
  );

  const targetRocPng = path.join(runDir, "roc_curve.png").replace(/\\/g, "/");
  const targetCmPng = path
    .join(runDir, "confusion_matrix.png")
    .replace(/\\/g, "/");
  const targetResultsJson = path
    .join(runDir, "auto_ML_results.json")
    .replace(/\\/g, "/");
  const targetRocJson = path
    .join(runDir, "roc_curve_data.json")
    .replace(/\\/g, "/");
  const targetResourcePng = path
    .join(runDir, "model_resource_comparison.png")
    .replace(/\\/g, "/");
  const plotsDir = path
    .join(__dirname, "thesis_pipeline/plots")
    .replace(/\\/g, "/");
  const targetPerformancePng = path
    .join(plotsDir, "model_performance_comparison.png")
    .replace(/\\/g, "/");
  const targetTimePng = path
    .join(plotsDir, "model_training_time_comparison.png")
    .replace(/\\/g, "/");

  const modelsMetricsMerged = {};
  const trainMetrics = trainResult.metrics || {};
  const trainDescriptions = trainResult.model_descriptions || {};
  for (const [modelName, metricsObj] of Object.entries(trainMetrics)) {
    modelsMetricsMerged[modelName] = {
      ...metricsObj,
      description: trainDescriptions[modelName] || "",
    };
  }

  const evalPrompt = evalPromptTemplate
    .replaceAll(
      "${models_trained}",
      JSON.stringify(trainResult.models_trained || trainResult.results || {}),
    )
    .replaceAll("${test_features_path}", featureResult.test_features_path)
    .replaceAll("${test_labels_path}", featureResult.test_labels_path)
    .replaceAll("${models_dir}", modelsDir)
    .replaceAll("${plots_dir}", plotsDir)
    .replaceAll("${roc_plot_path}", targetRocPng)
    .replaceAll("${cm_plot_path}", targetCmPng)
    .replaceAll("${performance_plot_path}", targetPerformancePng)
    .replaceAll("${time_plot_path}", targetTimePng)
    .replaceAll("${resource_plot_path}", targetResourcePng)
    .replaceAll("${results_json_path}", targetResultsJson)
    .replaceAll("${roc_json_path}", targetRocJson)
    .replaceAll("${models_metrics}", JSON.stringify(modelsMetricsMerged))
    .replaceAll("${mode}", pipelineMode);

  const evalGen = await generateTextWithRetry({
    model: google("gemini-2.5-flash"),
    prompt: evalPrompt,
  });

  console.log("⚡ LLM değerlendirme kodu üretti. Grafikler çiziliyor...");
  const evalOpRes = await mcp.callTool(
    {
      name: "evaluate_and_plot_tool",
      arguments: { code: evalGen.text },
    },
    undefined,
    { timeout: GLOBAL_TIMEOUT_MS },
  );

  const evalResult = evalOpRes.structuredContent;
  if (!evalResult || evalResult.error) {
    throw new Error(
      `Değerlendirme başarısız: ${evalResult?.error || "Bilinmeyen Hata"}`,
    );
  }

  // Mark step 4 as completed and save state
  state.step_4_eval = "completed";
  state.evalResult = evalResult;
  await saveState(runDir, state);

  // ─── Step 5: Tez Görselleştirmeleri (SHAP, Error Analysis, Word Cloud, t-SNE, Calibration) ───
  state.step_5_viz = state.step_5_viz || "pending";

  if (state.step_5_viz === "completed") {
    console.log("⏭️ Adım 5: Görselleştirme zaten tamamlanmış, atlanıyor...");
  } else {
    console.log(
      "\n🎨 Adım 5: Tez görselleştirme şablonu yükleniyor ve yerel olarak çalıştırılıyor (API kotaları harcanmaz)...",
    );

    const vizOutputDir = path.join(runDir, "viz_outputs").replace(/\\/g, "/");
    await fs.mkdir(vizOutputDir, { recursive: true });

    // Metrics JSON path from eval step
    const modelsMetricsJsonPath = targetResultsJson;

    const vizScriptTemplate = await fs.readFile(
      path.join(__dirname, "prompts/thesis_visualizer.py"),
      "utf8",
    );

    const vizCode = vizScriptTemplate
      .replaceAll("${mode}", pipelineMode)
      .replaceAll("${models_dir}", modelsDir)
      .replaceAll("${test_features_path}", featureResult.test_features_path)
      .replaceAll("${test_labels_path}", featureResult.test_labels_path)
      .replaceAll("${cleaned_dataset_path}", cleanResult.cleaned_dataset_path)
      .replaceAll("${vectorizer_pkl_path}", featureResult.vectorizer_pkl_path)
      .replaceAll("${models_metrics_json}", modelsMetricsJsonPath)
      .replaceAll("${viz_output_dir}", vizOutputDir);

    console.log("⚡ Hazır görselleştirme şablonu yerel python'a aktarılıyor...");
    const vizOpRes = await mcp.callTool(
      {
        name: "evaluate_and_plot_tool",
        arguments: { code: vizCode },
      },
      undefined,
      { timeout: GLOBAL_TIMEOUT_MS },
    );

    const vizResult = vizOpRes.structuredContent;
    if (!vizResult || vizResult.error) {
      console.warn(
        `⚠️ Görselleştirme adımı başarısız: ${
          vizResult?.error || "Bilinmeyen Hata"
        }. Pipeline devam ediyor...`,
      );
    } else {
      state.step_5_viz = "completed";
      state.vizResult = vizResult;
      await saveState(runDir, state);
      console.log("✅ Tez görselleştirmeleri başarıyla üretildi.");
      if (vizResult.shap_plots?.length > 0) {
        console.log("🔬 SHAP Grafikleri:", vizResult.shap_plots.join(", "));
      }
      if (vizResult.wordcloud_comparison_plot) {
        console.log("☁️ Kelime Bulutu:", vizResult.wordcloud_comparison_plot);
      }
      if (vizResult.tsne_plot) {
        console.log("🗺️ t-SNE Grafiği:", vizResult.tsne_plot);
      }
      if (vizResult.calibration_plot) {
        console.log("📐 Kalibrasyon Eğrisi:", vizResult.calibration_plot);
      }

      // Copy viz outputs to latest dir
      try {
        const vizLatestDir = path.join(latestDir, "viz_outputs");
        await fs.mkdir(vizLatestDir, { recursive: true });
        const vizFiles = await fs.readdir(vizOutputDir);
        for (const file of vizFiles) {
          await fs
            .copyFile(
              path.join(vizOutputDir, file),
              path.join(vizLatestDir, file),
            )
            .catch(() => {});
        }
        console.log(
          "✅ Görselleştirme dosyaları 'latest/viz_outputs/' klasörüne kopyalandı.",
        );
      } catch (copyErr) {
        console.warn("⚠️ viz_outputs kopyalama hatası:", copyErr.message);
      }
    }
  }

  // 📁 Tüm çıktıyı LATEST klasörüne senkronize et
  console.log("\n🔄 Sonuçlar 'latest/' klasörüne kopyalanıyor...");
  try {
    const files = await fs.readdir(latestDir);
    for (const file of files) {
      await fs.unlink(path.join(latestDir, file)).catch(() => {});
    }

    const runFiles = await fs.readdir(runDir);
    for (const file of runFiles) {
      await fs.copyFile(path.join(runDir, file), path.join(latestDir, file));
    }
    console.log("✅ 'latest/' başarıyla güncellendi.");
  } catch (copyErr) {
    console.warn("⚠️ Kopyalama işlemi sırasında uyarı:", copyErr.message);
  }

  // Calculate total execution time
  const totalDurationSeconds = Math.round(
    (Date.now() - state.pipeline_start_time) / 1000,
  );
  state.pipeline_total_duration_seconds = totalDurationSeconds;

  const hours = Math.floor(totalDurationSeconds / 3600);
  const minutes = Math.floor((totalDurationSeconds % 3600) / 60);
  const seconds = totalDurationSeconds % 60;
  state.pipeline_total_duration_formatted = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

  await saveState(runDir, state);

  console.log("\n🚀 [MODÜLER PIPELINE TAMAMLANDI] 🚀");
  console.log("📂 Deney Klasörü:", runDir);
  console.log(
    "⏱️ Toplam Çalışma Süresi:",
    state.pipeline_total_duration_formatted,
  );
  console.log("📈 ROC Grafiği:", evalResult.roc_plot_path);
  console.log("📉 Confusion Matrix Grafiği:", evalResult.cm_plot_path);
  if (evalResult.performance_plot_path) {
    console.log(
      "📊 Model Performans Karşılaştırma Grafiği:",
      evalResult.performance_plot_path,
    );
  }
  if (evalResult.time_plot_path) {
    console.log(
      "⏳ Model Eğitim Süresi Karşılaştırma Grafiği:",
      evalResult.time_plot_path,
    );
  }
  if (evalResult.resource_plot_path) {
    console.log(
      "🔋 Donanım Kaynak Tüketimi Karşılaştırma Grafiği:",
      evalResult.resource_plot_path,
    );
  }
  console.log("💾 Sonuç Metrikleri (Arşiv):", evalResult.results_json_path);
  console.log(
    "💾 Sonuç Metrikleri (Son):",
    path.join(latestDir, "auto_ML_results.json"),
  );
  if (evalResult.roc_json_path) {
    console.log("💾 ROC Eğrisi Verileri (Arşiv):", evalResult.roc_json_path);
    console.log(
      "💾 ROC Eğrisi Verileri (Son):",
      path.join(latestDir, "roc_curve_data.json"),
    );
  }

  // Run compare_ml_dl.py to generate cross-mode comparison report
  console.log("\n📊 ML vs DL karşılaştırma analizi tetikleniyor...");
  try {
    execSync("python compare_ml_dl.py", { stdio: "inherit" });
  } catch (err) {
    console.warn(
      "⚠️ ML vs DL karşılaştırma raporu üretilemedi (Henüz diğer mod tamamlanmamış olabilir):",
      err.message,
    );
  }
}

main();
