// client.js - MCP + Gemini AI Agent Entegrasyonu
import "dotenv/config";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { input, select } from "@inquirer/prompts";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { generateText, jsonSchema } from "ai";

// 1. MCP İstemcisini Hazırla
const mcp = new Client(
  { name: "phishing-agent", version: "1.0.0" },
  { capabilities: {} }
);

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
              } (Argümanlar: ${JSON.stringify(args)})`
            );
            return await mcp.callTool(
              {
                name: tool.name,
                arguments: args,
              },
              undefined,
              { timeout: 600000 } // <-- Buraya da eklersek sohbet modu da ölümsüz olur
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
        2
      )
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
        const result = await mcp.callTool({
          name: toolName,
          arguments: argsObj,
        });
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
                result.structuredContent ?? result.content ?? result
              ),
            },
          ],
        });

        console.log(
          "\n✅ Tool çağrısı başarılı. Agent'a sonuç geri bildirildi."
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

    // GLOBAL timeout'u 300 saniye yap
    // MCP timeouts override - DOĞRU ANAHTARLAR
    const TIMEOUT_MS = 600000; // 10 Dakika (600000 ms)

    // 1. Varsa mevcut ayarları koru, yoksa boş obje oluştur
    mcp.requestOptions = mcp.requestOptions || {};

    // 2. Timeout'u ZORLA ayarla (if kontrolü yok!)
    mcp.requestOptions.timeout = TIMEOUT_MS;

    // 3. _timeoutInfo (Internal) ayarlarını da ZORLA
    if (!mcp._timeoutInfo) {
      mcp._timeoutInfo = new Map();
    }
    mcp._timeoutInfo.set("callTool", { timeout: TIMEOUT_MS });
    mcp._timeoutInfo.set("tools/call", { timeout: TIMEOUT_MS }); // Her ihtimale karşı
    // -----------------------------

    // Agent'ın kullanabileceği araçları çek
    const [{ tools }] = await Promise.all([mcp.listTools()]);
    console.log(
      "🤖 Agent'ın Kullanılabilir Tool'ları:",
      tools.map((t) => t.name)
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
            })
          );
          const test_size = Number(
            await input({
              message: "Test Size (0.1 - 0.9):",
              default: "0.2",
            })
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
              { timeout: 600000 }
            );

            console.log("\n📊 Eğitim Sonucu:");
            console.log(
              JSON.stringify(
                result.structuredContent ?? result.content ?? result,
                null,
                2
              )
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

          const objective = await input({
            message: "Objective",
            initial: "Train deep learning models",
          });

          console.log("\n🚀 AutoDL agent çağrılıyor — LLM DL kodu üretecek...");

          try {
            const autoRes = await mcp.callTool(
              {
                name: "auto_dl_agent",
                arguments: { dataset_path, objective },
              },
              undefined,
              { timeout: 3600000 } // deep learning daha uzun sürebilir 1 saat
            );

            console.log("\n--- AutoDL Result ---");
            console.log(
              JSON.stringify(autoRes.structuredContent ?? autoRes, null, 2)
            );
            console.log("--- end ---\n");
          } catch (err) {
            console.error("AutoDL çağrısı başarısız:", err);
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
                2
              )
            );
          } catch (err) {
            console.error("❌ Tool hatası:", err);
          }
          break;
        }
        // main() içindeki switch'e ekle:
        case "AutoML": {
          // Kullanıcıdan kısa parametreler alalım (default'ları da kullanabilir)
          const dataset_path = await input({
            message: "Dataset yolu (enter ile varsayılan kullanılır)",
            initial: "Phishing_Email.csv",
          });

          const objective = await input({
            message: "Objective (ör. 'Maximize F1 score for phishing class')",
            initial: "Detect phishing emails with high F1 score",
          });

          console.log(
            "\n🚀 AutoML agent çağrılıyor — LLM kod üretecek ve modeli eğitecek. Bu biraz sürebilir..."
          );

          try {
            const autoRes = await mcp.callTool(
              {
                name: "auto_ml_agent",
                arguments: { dataset_path, objective, __timeout: 600000 },
              },
              undefined, // 👈 2. Argüman (Schema): Boş geçiyoruz (undefined)
              { timeout: 600000 } // 👈 3. Argüman (Options): Timeout BURAYA gelmeli!
            );

            console.log("\n--- AutoML Result (structuredContent) ---");
            console.log(
              JSON.stringify(autoRes.structuredContent ?? autoRes, null, 2)
            );
            console.log("--- end ---\n");
          } catch (err) {
            console.error("AutoML çağrısı başarısız:", err);
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

main();
