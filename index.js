require("dotenv").config();
const restify = require("restify");
const fetch = require("node-fetch");
const { BotFrameworkAdapter } = require("botbuilder");

// ----------------------------------------
// Bot Adapter (uses Azure Bot Registration)
// ----------------------------------------
const adapter = new BotFrameworkAdapter({
  appId: process.env.MICROSOFT_APP_ID,
  appPassword: process.env.MICROSOFT_APP_PASSWORD
});

// ----------------------------------------
// Create Bot Server
// ----------------------------------------
const server = restify.createServer();
server.listen(process.env.PORT || 3978, () => {
  console.log("🤖 Teams Bot running at http://localhost:3978/api/messages");
});

// ----------------------------------------
// Message Handler
// ----------------------------------------
server.post("/api/messages", async (req, res) => {
    try {
        await adapter.processActivity(req, res, async (context) => {

            if (context.activity.type === "message") {

                const userMsg = context.activity.text;
                const sessionId = context.activity.conversation.id;
                console.log("💬 User:", userMsg);
                console.log("🧠 Session ID:", sessionId);
                try {
                    const apiRes = await fetch(
                        process.env.FLASK_SERVER_URL + "/agent/chat",
                        {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },

                            body: JSON.stringify({
                                session_id: sessionId,
                                message: userMsg
                            })

                        }
                    );

                    const data = await apiRes.json();
                    const reply = data.answer || "⚠️ No answer returned.";

                    await context.sendActivity(reply);

                } catch (err) {
                    console.error("❌ Backend error:", err);
                    await context.sendActivity("⚠️ Error contacting backend.");
                }

            }
        });

    } catch (err) {
        console.error("❌ processActivity failed:", err);
    }
});

