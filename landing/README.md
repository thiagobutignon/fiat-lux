# Fiat Lux Landing Page

Modern, responsive landing page for Fiat Lux - Universal Grammar Engine.

## 🚀 Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS framework
- **React 18** - Latest React features

## 📂 Project Structure

```
landing/
├── app/
│   ├── globals.css          # Global styles with Tailwind
│   ├── layout.tsx            # Root layout
│   └── page.tsx              # Home page
├── components/
│   └── sections/
│       ├── Hero.tsx          # Hero section with stats
│       ├── Features.tsx      # Features showcase
│       ├── Demo.tsx          # Interactive demo
│       ├── Architecture.tsx  # Clean Architecture visualization
│       ├── CodeExample.tsx   # Code examples with tabs
│       └── Footer.tsx        # Footer with links
├── public/                   # Static assets
├── next.config.js            # Next.js configuration
├── tailwind.config.ts        # Tailwind configuration
├── tsconfig.json             # TypeScript configuration
└── package.json              # Dependencies
```

## 🎨 Features

- ✨ **Responsive Design** - Works on all devices
- 🌓 **Dark Mode Support** - Automatic dark mode
- ⚡ **Performance Optimized** - Fast loading times
- 🎯 **Interactive Demo** - Real-time code examples
- 📊 **Architecture Visualization** - Clean Architecture showcase
- 🎨 **Modern UI** - Tailwind CSS with custom animations

## 🛠️ Development

### Install Dependencies

```bash
cd landing
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the landing page.

### Build for Production

```bash
npm run build
npm start
```

## 📝 Sections

### 1. Hero
- Eye-catching hero with animated gradient text
- Key metrics (77 tests, 5ms execution, 99% cache rate)
- CTA buttons for demo and GitHub

### 2. Features
- 6 feature cards with icons
- Clean, modern card design
- Claude Code attribution

### 3. Demo
- Interactive code editor
- Real-time validation example
- Performance metrics display

### 4. Architecture
- Visual folder structure
- Layer-by-layer explanation
- Benefits of Clean Architecture

### 5. Code Examples
- Tabbed interface (Basic, Custom, Pattern Loader)
- Syntax-highlighted code blocks
- Quick installation guide

### 6. Footer
- Links to documentation
- Social media links
- Acknowledgements
- MIT License info

## 🎨 Customization

### Colors

Edit `tailwind.config.ts` to customize colors:

```typescript
theme: {
  extend: {
    colors: {
      primary: '#6366f1',    // Indigo
      secondary: '#8b5cf6',  // Purple
    },
  },
}
```

### Content

- **Hero stats**: Edit `components/sections/Hero.tsx`
- **Features**: Edit `components/sections/Features.tsx`
- **Code examples**: Edit `components/sections/CodeExample.tsx`

## 🏆 Benchmark Results

We built a comprehensive benchmark comparing **deterministic grammar-based systems** vs **probabilistic AI/ML approaches** for trading signal generation.

### Quick Results (100 test cases on Mac M4 Pro)

| System | Accuracy | Latency | Cost | Explainable |
|--------|----------|---------|------|-------------|
| **Grammar Engine** | **100%** | **0.013ms** | **$0.00** | ✅ **100%** |
| Custom LSTM | 56% | 45.9ms | $0.00 | ❌ 0% |
| llama.cpp (8B Q4) | 48% | 3.5s | $0.00 | ❌ 0% |
| Fine-tuned Llama 70B | 30% | 121ms | $0.00 | ❌ 0% |
| GPT-4 | 27% | 350ms | $0.05 | ❌ 0% |
| Claude 3.5 Sonnet | 26% | 280ms | $0.04 | ❌ 0% |

### Key Findings

- **Grammar Engine is 269,695x faster** than local LLMs (llama.cpp)
- **100% accuracy** on candlestick pattern detection
- **100% explainable** - every decision is rule-based
- **$0 cost** - no API calls required
- **Perfect reproducibility** - same input always produces same output

### Run the Benchmark

```bash
# Quick test (100 cases, ~5 minutes)
npm run benchmark:quick

# Full benchmark (1000 cases, ~30 minutes)
npm run benchmark:full

# Mac M-series setup (recommended)
See MAC_SETUP.md for llama.cpp installation
```

**Full documentation**: See `BENCHMARK_README.md` for setup and detailed results.

## 📄 License

MIT License - Built with [Claude Code](https://claude.com/claude-code) by Anthropic.

## 🤖 Generated with Claude Code

This landing page was built using Claude Code, demonstrating the power of AI-assisted development following Clean Architecture principles.
