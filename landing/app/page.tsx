import Hero from '@/components/sections/Hero'
import Features from '@/components/sections/Features'
import Demo from '@/components/sections/Demo'
import Architecture from '@/components/sections/Architecture'
import CodeExample from '@/components/sections/CodeExample'
import Footer from '@/components/sections/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Hero />
      <Features />
      <Demo />
      <Architecture />
      <CodeExample />
      <Footer />
    </main>
  )
}
