# Bark TTS Flask App - Vercel Deployment

This guide contains specific instructions for deploying this Bark TTS application to Vercel.

## Optimized for Vercel

This version of the application has been specially optimized for Vercel deployment:

1. **Modular Architecture**: The code is split into multiple smaller files to avoid Vercel's lambda function size limits.
2. **Environment Configuration**: Special settings for AWS Lambda (which Vercel uses) are automatically applied.
3. **Memory Optimization**: Thread control and memory management to work within Vercel's limits.
4. **Model Caching**: Uses Vercel's `/tmp` directory for model caching.

## Deployment Steps

1. **Push to GitHub**:
   - Push this repository to GitHub

2. **Set Up Vercel**:
   - Sign up for Vercel (if you haven't already): https://vercel.com/signup
   - Install the Vercel CLI: `npm i -g vercel`
   - Login to Vercel: `vercel login`

3. **Deploy the Project**:
   ```
   vercel
   ```

4. **Environment Variables**:
   Set the following environment variables in the Vercel project settings:
   - `FLASK_ENV=production`
   - `OMP_NUM_THREADS=1`
   - `TORCH_USE_RTLD_GLOBAL=1`

## Important Notes

- **Cold Starts**: The first request after deployment may take a while as models are loaded.
- **Function Size**: The full Bark model is large. If you encounter function size limits, you might need to explore Vercel's Enterprise plan or consider alternative hosting options like a dedicated server or cloud VMs.
- **Runtime Limit**: Vercel's serverless functions have time limits (usually around 10 seconds for free tier). Audio generation might exceed this time for long texts.

## Troubleshooting

If you encounter issues, check:

1. **Logs**: View logs in the Vercel dashboard for error messages.
2. **Function Memory**: You might need to increase function memory in Vercel project settings.
3. **Deployment Size**: Make sure all required files are being included in the deployment.

## Alternative Deployment Options

If Vercel's limitations are too restrictive, consider:

- **Railway**: Similar to Vercel but with more generous resource limits
- **DigitalOcean App Platform**: Supports Flask apps natively
- **Google Cloud Run**: Good for containerized Flask applications
- **Heroku**: Traditional choice for Flask applications

For high-performance needs, a dedicated server or VM is recommended as Bark is resource-intensive. 