// server.js
// This backend uses Node.js and Express to handle AI summarization and email sharing.
// It exposes two endpoints: /summarize and /share.

// 1. Setup and Dependencies
// To run this code, make sure you have Node.js installed.
// Then, install the required packages by running:
// npm install express cors body-parser nodemailer dotenv

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');

// Load environment variables from a .env file
dotenv.config();

const app = express();
const port =  process.env.PORT || 3001;; // Choose a port for your backend

// Use environment variables for sensitive data in a real-world application.
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const EMAIL_SERVICE_USER = process.env.EMAIL_SERVICE_USER;
const EMAIL_SERVICE_PASS = process.env.EMAIL_SERVICE_PASS;

// Middleware to parse JSON and enable CORS
app.use(bodyParser.json());
app.use(cors());

// Nodemailer transporter setup
const transporter = nodemailer.createTransport({
    service: 'gmail', // Example using Gmail, requires "App Passwords"
    auth: {
        user: EMAIL_SERVICE_USER,
        pass: EMAIL_SERVICE_PASS,
    }
});

// Helper function to call either the Groq or OpenAI API
async function callLLMAPI(prompt, text, provider) {
    let apiUrl, headers, payload;

    if (provider === 'groq') {
        apiUrl = 'https://api.groq.com/openai/v1/chat/completions';
        headers = {
            'Authorization': `Bearer ${GROQ_API_KEY}`,
            'Content-Type': 'application/json'
        };
        payload = {
            model: 'llama3-8b-8192',
            messages: [{ role: "user", content: prompt + '\n\n' + text }],
            temperature: 0.5,
        };
    } else if (provider === 'openai') {
        apiUrl = 'https://api.openai.com/v1/chat/completions';
        headers = {
            'Authorization': `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
        };
        payload = {
            model: 'gpt-3.5-turbo', // A good default model
            messages: [{ role: "user", content: prompt + '\n\n' + text }],
            temperature: 0.5,
        };
    } else {
        throw new Error('Invalid AI provider specified.');
    }
    
    // Exponential backoff for API calls
    const maxRetries = 5;
    let retries = 0;
    
    while (retries < maxRetries) {
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                if (response.status === 429) {
                    const delay = Math.pow(2, retries) * 1000;
                    console.error(`Rate limit exceeded. Retrying in ${delay / 1000} seconds...`);
                    await new Promise(res => setTimeout(res, delay));
                    retries++;
                    continue;
                }
                throw new Error(`API call failed with status: ${response.status} from ${provider} API`);
            }

            const result = await response.json();
            
            const content = result?.choices?.[0]?.message?.content;
            if (content) {
                return content;
            } else {
                throw new Error(`Invalid API response from ${provider}: No content found.`);
            }
        } catch (error) {
            console.error(`Error calling ${provider} API:`, error);
            const delay = Math.pow(2, retries) * 1000;
            console.error(`An error occurred. Retrying in ${delay / 1000} seconds...`);
            await new Promise(res => setTimeout(res, delay));
            retries++;
        }
    }
    
    throw new Error(`Failed to generate summary after multiple retries with ${provider}.`);
}

// Function to split large text into chunks and summarize them
async function processLargeTranscript(transcript, prompt) {
    const maxTokensPerChunk = 8000;
    const words = transcript.split(/\s+/);
    const chunks = [];
    let currentChunk = '';

    for (const word of words) {
        if ((currentChunk + ' ' + word).split(/\s+/).length > maxTokensPerChunk) {
            chunks.push(currentChunk);
            currentChunk = '';
        }
        currentChunk += (currentChunk ? ' ' : '') + word;
    }
    if (currentChunk) {
        chunks.push(currentChunk);
    }
    
    const partialSummaries = await Promise.all(chunks.map(chunk =>
        callLLMAPI(`Summarize the following text to extract all key information and main points. Do not lose any important details.`, chunk, 'groq')
    ));
    
    const combinedSummary = partialSummaries.join('\n\n---\n\n');
    const finalPrompt = `I have several summaries of different parts of a long document. Combine them into a single, cohesive summary. Then, apply the following instruction: ${prompt} The final summary should be free of any spelling mistakes. Do not include any introductory phrases like "Here is a summary:" or "This is a summary of the text:".`;
    
    const finalSummary = await callLLMAPI(finalPrompt, combinedSummary, 'groq');
    return finalSummary;
}

// Top-level function with a fallback
async function generateSummaryWithFallback(transcript, prompt) {
    const finalPrompt = `${prompt} Ensure the output is free of spelling and grammatical errors. Do not include any introductory phrases like "Here is a summary:" or "This is a summary of the text:".`;

    try {
        const summary = await callLLMAPI(finalPrompt, transcript, 'groq');
        return summary;
    } catch (groqError) {
        console.error('Groq API failed. Attempting with OpenAI as a fallback...', groqError);
        try {
            const summary = await callLLMAPI(finalPrompt, transcript, 'openai');
            return summary;
        } catch (openaiError) {
            console.error('OpenAI API also failed.', openaiError);
            throw new Error('Failed to generate summary with both Groq and OpenAI APIs.');
        }
    }
}

// 2. API Endpoints

// Endpoint to generate a summary
app.post('/summarize', async (req, res) => {
    try {
        const { transcript, prompt } = req.body;
        
        if (!transcript || !prompt) {
            return res.status(400).json({ error: 'Transcript and prompt are required.' });
        }

        const maxChunkLength = 8000 * 4;
        let summary;

        if (transcript.length > maxChunkLength) {
            summary = await processLargeTranscript(transcript, prompt);
        } else {
            summary = await generateSummaryWithFallback(transcript, prompt);
        }
        
        res.status(200).json({ summary });

    } catch (error) {
        console.error('Error generating summary:', error);
        res.status(500).json({ error: 'Failed to generate summary.' });
    }
});

// Endpoint to share the summary via email
app.post('/share', async (req, res) => {
    try {
        const { summary, recipients } = req.body;
        
        if (!summary || !recipients || recipients.length === 0) {
            return res.status(400).json({ error: 'Summary and at least one recipient are required.' });
        }

        const mailOptions = {
            from: EMAIL_SERVICE_USER,
            to: recipients.join(','),
            subject: 'Meeting Notes Summary',
            text: `Hello,\n\nHere is the summary of the meeting notes:\n\n${summary}`,
            html: `<h3>Meeting Notes Summary</h3><p>Hello,</p><p>Here is the summary of the meeting notes:</p><p>${summary.replace(/\n/g, '<br>')}</p>`,
        };

        await transporter.sendMail(mailOptions);
        res.status(200).json({ message: 'Summary shared successfully.' });

    } catch (error) {
        console.error('Error sharing summary:', error);
        res.status(500).json({ error: 'Failed to share summary.' });
    }
});


// 3. Start the server
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
});