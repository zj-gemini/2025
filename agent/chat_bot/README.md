# Customer FAQ System

## Project Overview
Build a customer FAQ system that integrates AI with knowledge base support and ticket management functionality.
The system should provide a chat-based UI for users to interact with an AI agent, with fallback task creation when the AI cannot provide adequate responses.



---

## Core Requirements

### 1. Knowledge Base Integration
- Implement a knowledge base system that can be accessible by AI
- AI should attempt to answer user questions using the knowledge base
- Knowledge base can contain FAQs, product information, etc, expected to be small text file (<500k bytes), so database may not be required
- You can find a few pages (say 10 URLs) and save to a text file as knowledge base

---

### 2. AI Chat Interface
- Create a chat-based UI similar to existing customer service chat products
- AI responds based on knowledge base content
- Chat history should be maintained during the session
- Any LLM is totally fine (such as ChatGPT API, or local solution), the criteria is the capability of LLM integration, not the LLM quality itself.

---

### 3. Ticket Generation
- When AI cannot provide a satisfactory answer, automatically generate a ticket
- Tasks should be stored in the database with relevant information (user question, timestamp, user contact, etc.)
- Basic page to display generated tickets in DB, no need fancy UI

---

## Technical Stack Suggestions
- Frontend: Vue.js or React
- Backend: Python FastAPI or Go
- Database: Any relational DB (PostgreSQL, MySQL) or key-value store (Redis, etc.)

> Note: These are suggestions only. Use any technology stack you're already familiar with.

---

## Testing Requirements
- End-to-end testing for core business logic
- Test the complete flow: user question → AI response → task generation (if needed)
- Unit tests for core functions are enough, no need high coverage
- Simulation testing if more relevant than traditional testing

---

## Deliverables

### Code Submission
- Check in complete code to GitHub repository
- Repository access to be shared with reviewer: **gensparkreviewer@gmail.com**
- README.md with setup instructions in local run (recommended), no need cloud deploy
- Screenshots or screen recording (optional but appreciated)

---

## Extra Bonus
Feel free to add any interesting features which you think would help in the real scenario.

---

## Review Process

### Review Meeting Preparation
- Prepare bullet points explaining your technical decisions and architecture choices
- Be ready to demonstrate the application's core functionality, and testing results
- No comprehensive documentation required - focus on key decision rationale

---

## Evaluation Criteria
- Functionality: All core features working as specified
- Testing: Adequate test coverage of critical functionality
- Technical Decisions: Ability to explain implementation choices and code

---

## Demo

[High-level Architecture](https://docs.google.com/document/d/1PAC0q6Ehvr62mtfZgQiUC_jh_JS_CdyTJPwfzKjJ3so/edit?tab=t.0#heading=h.arofyvvgmrql)

[Demo Recording](https://drive.google.com/file/d/1JIbL2hnRJfQa1VsA7_0v11hdcNP0Sjze/view?usp=sharing)

---

## Deployment

### Backend (FastAPI)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Set Gemini API Key:**
    Add below line to your .bashrc:
    ```
    export GOOGLE_API_KEY=[YOUR_GEMINI_API_KEY]
    ```
3.  **Run the Server:**
    The backend server will run on `http://localhost:8000`.
    ```bash
    uvicorn src.agent.main:app --host 0.0.0.0 --port 8000
    ```
    > **Note:** If you get an `address already in use` error, stop the existing process with:
    > ```bash
    > pkill -f uvicorn
    > ```

3.  **Deploying Changes:**
    To apply changes to the backend, you need to restart the server. You can do this by stopping the current server (if it's running) and starting it again using the commands above.

### Frontend (React)

1.  **Install Dependencies:**
    ```bash
    npm install --prefix src/ui
    ```

    Install latest Node.js
    ```
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    
    # Re-start terminal to activate env
    nvm install 22.12.0
    nvm use 22.12.0
    nvm alias default 22.12.0
    ```

2.  **Run the Development Server:**
    The frontend development server will run on `http://localhost:5173`.
    ```bash
    npm run dev --prefix src/ui
    ```
    > **Note:** If the port is already in use, stop the existing process with the following command:
    > ```bash
    > pkill -f vite
    > ```
    > After stopping the process, you can try running the server again.

---

📧 **Email:** [gensparkreviewer@gmail.com](mailto:gensparkreviewer@gmail.com)
