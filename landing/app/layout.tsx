import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Fiat Lux - Universal Grammar Engine',
  description: 'A generic, configurable grammar engine that validates and auto-repairs structured data based on customizable grammatical rules.',
  keywords: ['grammar', 'validation', 'auto-repair', 'clean-architecture', 'typescript'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
