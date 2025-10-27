
README — Prompt Flow Chatbot (VS Code + Azure OpenAI)

This project is a small chatbot built with Azure’s Prompt Flow and run locally in VS Code. You need an Azure OpenAI resource (your endpoint looks like https://<resource>.openai.azure.com) and at least one deployment (its Deployment name, e.g., gpt-4o or gpt-4-1). In VS Code, use your project’s .venv as the interpreter and install: promptflow, promptflow-tools, and promptflow-azure. Create a local connection named aoai_chat that points to your resource endpoint and uses API version 2024-02-15-preview. In the Prompt Flow panel, open Connections → Azure OpenAI → Create from YAML, select your YAML (with api_key: "<user-input>"), and paste your key when prompted. On macOS, Keychain may ask for your Mac login password; allow it so runs succeed. Open the flow (flow.dag.yaml), mark question as chat input, chat_history as chat history, wire the LLM node’s output to answer, set connection = aoai_chat, and set deployment_name to the exact name you used in Azure OpenAI → Deployments. You can now run locally with pf flow test --flow /absolute/path/to/flow-folder or use the Chat button in the Prompt Flow panel.

README — Updated Changes (Changelog)

Keep recent changes here in plain language so future you understands what changed and why. For example: “Switched deployment_name from gpt-35-turbo to gpt-4o to fix 404 errors,” “Moved API base to cpt-azureopenai resource in eastus to match deployment,” “Installed promptflow-tools in .venv to resolve EmptyLLMApiMapping,” “Changed chat_history input type to list to enable threaded chat,” or “Added temperature and response formatting to the LLM node for more concise replies.” Each change should say the reason and the file/node impacted so it’s easy to trace later.

Components — What Each Part Is

The flow is the blueprint describing how messages move through your app. The inputs are the pieces the user and chat session provide (question and chat_history), and the outputs are what you return (answer). The LLM node is the model call that turns your prompt into a reply. The connection stores your endpoint, API version, and key so the LLM node can talk to Azure OpenAI. The prompt template (system message and formatting) sets the bot’s behavior and how chat history is shown to the model. Any Python tools you add (optional) are little functions the flow can call for retrieval or custom logic. Finally, the run/test is how you execute the flow locally, either from VS Code’s Chat panel or with pf flow test.

Front End vs. Back End — How This Project Splits

The front end in this repo is minimal: it’s the Prompt Flow Chat UI inside VS Code used for testing (or any simple UI you wire up later). It’s mainly about sending the user’s text, keeping the chat history, and showing the model’s response. The back end is the flow itself plus the Azure OpenAI call. The back end prepares the prompt, includes relevant history, calls the deployed model in your Azure resource, and returns the resulting text to the UI. If you later deploy a managed endpoint, the back end will also expose an HTTP endpoint that a real web app can call.

High-Level Explanation — How It Works

When you type a message, the flow collects your text and the running conversation, injects both into the model prompt, and calls your Azure OpenAI deployment using the connection you configured. The model replies with the next message. The flow saves that reply as answer and adds it to the conversation so the next turn stays in context. Nothing fancy is required: the critical parts are using the correct endpoint, API version, and deployment name, and making sure your local Python has the Prompt Flow packages. If any of those are off, the run will fail until they match.

Low-Level Technical Notes — What Matters Under the Hood

The connection must include name, type: azure_open_ai, api_base without a trailing slash, api_version set to 2024-02-15-preview, and an API key. In the VS Code UI route, keep api_key: "<user-input>" so the extension prompts you and saves the secret in Keychain; in the CLI route, you can put the key into a file you immediately delete after creation. The LLM node requires the connection name and the exact deployment_name you created in Azure (dashes vs dots must match exactly). The inputs should be typed as string for question and list for chat_history if you want native chat behavior; the output should map to the LLM node’s response (e.g., answer). Your local run uses the Python interpreter you selected in VS Code; if promptflow-tools isn’t installed there, you’ll see “EmptyLLMApiMapping.” If the model name is wrong or in a different resource/region, you’ll get DeploymentNotFound (404). If the connection YAML is created from a temporary template rather than the Connections panel or CLI, you’ll see “Not all required fields filled” until you create it correctly.

Quick Troubleshooting (kept short)

If you see “Not all required fields filled”, create the connection from the Connections panel using your saved YAML (not the temp file), and ensure api_base and api_version are present and valid. If you see “EmptyLLMApiMapping”, install promptflow-tools in your .venv and re-select that interpreter in VS Code. If you see DeploymentNotFound (404), copy the Deployment name verbatim from Azure OpenAI → Deployments and set it on the LLM node, and ensure your connection’s endpoint is the same resource/region that hosts that deployment. If macOS asks for Keychain access, use your Mac login password and choose Always Allow so runs don’t pause.

Running and Next Steps

Run locally from the terminal with pf flow test --flow /absolute/path/to/your/flow-folder or use the Chat in the Prompt Flow panel. If you later want a real web app, deploy the flow as a managed online endpoint in Azure Studio and call it from a tiny FastAPI or frontend app. Keep this README updated in the “Updated Changes” section so future you (or a teammate) can follow the thread of what changed and why.




Tutorial:

You need three things: a model deployed in your Azure OpenAI resource, a local Python env, and a Prompt Flow chat flow. In VS Code, pick your project’s venv as the interpreter and install the bits: pip install -U promptflow promptflow-tools promptflow-azure. Create the Azure OpenAI connection once. The easiest way is a tiny YAML file that says the connection name, your resource endpoint, and the API version. Use this form: name aoai_chat, type azure_open_ai, api_base https://<your-resource>.openai.azure.com, api_version 2024-02-15-preview. In the Prompt Flow panel choose Connections → “Create from YAML,” select that file, and when macOS asks for Keychain access, enter your Mac login password (you can “Always Allow”). If you prefer CLI, put the same fields plus your key into aoai_conn.yaml and run pf connection create file aoai_conn.yaml, then delete the file.

Open your flow in VS Code. Mark question as the chat input and chat_history as chat history; map the LLM node’s output to answer. On the LLM node set the connection to aoai_chat and set deployment_name to the exact Deployment name you see under Azure OpenAI -> Deployments (e.g., gpt-4o or gpt-4-1; it must match exactly). Now run locally with pf flow test flow /absolute/path/to/your/flow-folder or use the Chat button in the Prompt Flow panel.

If you get “EmptyLLMApiMapping,” you installed packages in a different Python—switch VS Code to your venv and pip install promptflow-tools. If you see “Not all required fields filled,” you likely tried to create the connection from the temp YAML—use Connections → Create from YAML on your saved file. If you see DeploymentNotFound (404), the deployment name doesn’t match what’s in Azure or the connection points to the wrong resource—fix the name or endpoint and rerun. That’s it.
