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
            return await mcp.callTool({ name: tool.name, arguments: args });
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
          { name: "Tools'u Listele/Kullan", value: "Tools" },
          { name: "Çıkış", value: "Exit" },
        ],
      });

      switch (option) {
        case "Query":
          await handleQuery(tools);
          break;
        case "Tools":
          // Tool'ları listeleme mantığını buraya eklenebilir
          console.log(
            "Kullanılabilir Tool'lar:",
            tools.map((t) => t.name)
          );
          break;
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
            const autoRes = await mcp.callTool({
              name: "auto_ml_agent",
              arguments: { dataset_path, objective },
            });

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
