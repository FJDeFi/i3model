"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Sidebar } from "@/components/sidebar"
import { ProjectCard } from "@/components/project-card"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Plus,
  Search,
  TrendingUp,
  Users,
  DollarSign,
  Target,
  Star,
  BarChart3,
  RefreshCw,
  Settings,
  Pin,
  Calendar,
} from "lucide-react"
import Link from "next/link"

// Mock data for demonstration
const mockProjects = [
  {
    id: "1",
    name: "Smart Home Entertainment Analysis",
    category: "Smart Home > Home Entertainment",
    createdAt: "2024-01-15",
    lastUpdated: "2024-01-20",
    status: "completed",
  },
  {
    id: "2",
    name: "Kitchen Appliances Market Research",
    category: "Home & Kitchen > Small Appliances",
    createdAt: "2024-01-10",
    lastUpdated: "2024-01-18",
    status: "completed",
  },
]

const mockCharts = [
  {
    id: "1",
    title: "Price Distribution Analysis",
    projectName: "Smart Home Entertainment Analysis",
    lastUpdated: "2024-01-20",
    autoUpdate: "weekly",
    type: "bar",
  },
  {
    id: "2",
    title: "Customer Sentiment Trends",
    projectName: "Kitchen Appliances Market Research",
    lastUpdated: "2024-01-18",
    autoUpdate: "monthly",
    type: "line",
  },
  {
    id: "3",
    title: "Market Share Evolution",
    projectName: "Smart Home Entertainment Analysis",
    lastUpdated: "2024-01-19",
    autoUpdate: "weekly",
    type: "pie",
  },
  {
    id: "4",
    title: "Feature Adoption Rates",
    projectName: "Kitchen Appliances Market Research",
    lastUpdated: "2024-01-17",
    autoUpdate: null,
    type: "bar",
  },
]

const examplePrompts = [
  { icon: TrendingUp, text: "Analyze the competitive landscape", color: "bg-blue-50 text-blue-700" },
  { icon: Users, text: "What are the top customer pain points?", color: "bg-green-50 text-green-700" },
  { icon: DollarSign, text: "Show me pricing trends in this category", color: "bg-purple-50 text-purple-700" },
  { icon: Target, text: "Identify market opportunities", color: "bg-orange-50 text-orange-700" },
  { icon: Star, text: "Compare feature preferences", color: "bg-pink-50 text-pink-700" },
  { icon: BarChart3, text: "Generate customer persona insights", color: "bg-indigo-50 text-indigo-700" },
]

export default function HomePage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const mostRecentProject = mockProjects[0]

  const getChartIcon = (type: string) => {
    switch (type) {
      case "line":
        return TrendingUp
      case "pie":
        return Target
      default:
        return BarChart3
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        projects={mockProjects}
        charts={mockCharts}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-gray-900">Xenith</h1>
          </div>

          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm" className="w-10 h-10 p-0">
              <Search className="w-4 h-4 text-gray-400" />
            </Button>
            <Link href="/onboarding">
              <Button>
                <Plus className="w-4 h-4 mr-1" />
                Project
              </Button>
            </Link>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          {mostRecentProject ? (
            <div className="max-w-6xl mx-auto space-y-6">
              {/* Most Recent Project */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">Current Project</h2>
                </div>

                <ProjectCard project={mostRecentProject} featured={true} />
              </div>

              {/* Subscribed Charts */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Subscribed Charts</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {mockCharts.map((chart) => {
                    const ChartIcon = getChartIcon(chart.type)
                    return (
                      <Card key={chart.id} className="hover:shadow-md transition-shadow">
                        <CardHeader className="pb-3">
                          <div className="flex items-start justify-between">
                            <div className="space-y-2">
                              <CardTitle className="text-base line-clamp-2">{chart.title}</CardTitle>
                              <div className="text-sm text-gray-500">{chart.projectName}</div>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
                                <Pin className="w-3 h-3" />
                              </Button>
                              <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
                                <Settings className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        </CardHeader>

                        <CardContent className="pt-0">
                          {/* Chart Visualization */}
                          <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg h-48 flex items-center justify-center mb-4 relative overflow-hidden">
                            <div className="absolute inset-0 bg-white/20"></div>
                            <div className="text-center text-gray-600 z-10">
                              <ChartIcon className="w-16 h-16 mx-auto mb-3 text-blue-600" />
                              <div className="text-sm font-medium">{chart.title}</div>
                              <div className="text-xs text-gray-500 mt-1">Live {chart.type} visualization</div>
                            </div>
                            {/* Simulated chart elements */}
                            <div className="absolute bottom-4 left-4 right-4">
                              <div className="flex items-end space-x-1 h-8">
                                {[...Array(8)].map((_, i) => (
                                  <div
                                    key={i}
                                    className="bg-blue-400/60 rounded-t flex-1"
                                    style={{ height: `${Math.random() * 100 + 20}%` }}
                                  ></div>
                                ))}
                              </div>
                            </div>
                          </div>

                          {/* Chart Info */}
                          <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
                            <div className="flex items-center space-x-1">
                              <Calendar className="w-3 h-3" />
                              <span>Updated {new Date(chart.lastUpdated).toLocaleDateString()}</span>
                            </div>
                            {chart.autoUpdate && (
                              <Badge variant="outline" className="text-xs">
                                <RefreshCw className="w-2 h-2 mr-1" />
                                Auto-updates {chart.autoUpdate}
                              </Badge>
                            )}
                          </div>

                          {/* Actions */}
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <ChartIcon className="w-4 h-4 text-gray-400" />
                              <span className="text-sm text-gray-600 capitalize">{chart.type} Chart</span>
                            </div>

                            <div className="flex items-center space-x-2">
                              <Button variant="outline" size="sm">
                                <RefreshCw className="w-3 h-3 mr-1" />
                                Update
                              </Button>
                              <Button variant="default" size="sm">
                                View Details
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <div className="mb-6">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <BarChart3 className="w-8 h-8 text-blue-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to Xenith</h2>
                  <p className="text-gray-600">
                    Start your first market research analysis to unlock powerful insights from Amazon product data.
                  </p>
                </div>

                <Link href="/onboarding">
                  <Button size="lg" className="mb-6">
                    <Plus className="w-4 h-4 mr-2" />
                    Start Your First Analysis
                  </Button>
                </Link>

                <div className="text-left">
                  <h3 className="font-medium text-gray-900 mb-3">What you can analyze:</h3>
                  <ul className="text-sm text-gray-600 space-y-2">
                    <li>• Competitive landscape and market positioning</li>
                    <li>• Customer pain points and preferences</li>
                    <li>• Pricing trends and opportunities</li>
                    <li>• Product feature analysis</li>
                    <li>• Market gaps and opportunities</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
