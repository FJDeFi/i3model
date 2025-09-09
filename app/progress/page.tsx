"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Database, Search, BarChart3, Users, CheckCircle, Clock, ArrowRight } from "lucide-react"
import { useRouter } from "next/navigation"

const analysisSteps = [
  {
    id: "collecting",
    title: "Collecting Product Listings",
    description: "Scanning Amazon for products in Smart Home > Home Entertainment",
    icon: Search,
    duration: 5000,
  },
  {
    id: "extracting",
    title: "Extracting Specifications",
    description: "Processing product details, features, and technical specifications",
    icon: Database,
    duration: 4000,
  },
  {
    id: "reviews",
    title: "Processing Customer Reviews",
    description: "Analyzing customer feedback, ratings, and sentiment patterns",
    icon: Users,
    duration: 6000,
  },
  {
    id: "analyzing",
    title: "Generating Market Insights",
    description: "Creating competitive analysis and market opportunity reports",
    icon: BarChart3,
    duration: 3000,
  },
]

export default function ProgressPage() {
  const [currentStep, setCurrentStep] = useState(0)
  const [progress, setProgress] = useState(0)
  const [completed, setCompleted] = useState(false)
  const [activityLog, setActivityLog] = useState<string[]>([])
  const router = useRouter()

  useEffect(() => {
    const totalDuration = analysisSteps.reduce((sum, step) => sum + step.duration, 0)
    let elapsed = 0

    const interval = setInterval(() => {
      elapsed += 100

      // Calculate overall progress
      const overallProgress = Math.min((elapsed / totalDuration) * 100, 100)
      setProgress(overallProgress)

      // Determine current step
      let stepElapsed = 0
      let newCurrentStep = 0
      for (let i = 0; i < analysisSteps.length; i++) {
        if (elapsed <= stepElapsed + analysisSteps[i].duration) {
          newCurrentStep = i
          break
        }
        stepElapsed += analysisSteps[i].duration
        newCurrentStep = i + 1
      }

      if (newCurrentStep !== currentStep) {
        setCurrentStep(newCurrentStep)

        // Add activity log entries
        if (newCurrentStep > 0 && newCurrentStep <= analysisSteps.length) {
          const completedStep = analysisSteps[newCurrentStep - 1]
          setActivityLog((prev) => [...prev, `✓ ${completedStep.title} completed`])
        }

        if (newCurrentStep < analysisSteps.length) {
          const currentStepData = analysisSteps[newCurrentStep]
          setActivityLog((prev) => [...prev, `→ Starting ${currentStepData.title.toLowerCase()}...`])
        }
      }

      // Add periodic updates
      if (elapsed % 2000 === 0 && newCurrentStep < analysisSteps.length) {
        const messages = [
          "Analyzing 1,250 products in Smart Home category",
          "Processing product specifications and features",
          "Extracting customer review sentiment data",
          "Identifying competitive pricing patterns",
          "Mapping feature preferences across segments",
        ]
        const randomMessage = messages[Math.floor(Math.random() * messages.length)]
        setActivityLog((prev) => [...prev, `• ${randomMessage}`])
      }

      if (elapsed >= totalDuration) {
        setCompleted(true)
        setActivityLog((prev) => [...prev, "✓ Market analysis complete! Generating insights..."])
        clearInterval(interval)
      }
    }, 100)

    return () => clearInterval(interval)
  }, [currentStep])

  const handleViewResults = () => {
    router.push("/project/1")
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Market Analysis in Progress</h1>
              <p className="text-sm text-gray-600">Smart Home &gt; Home Entertainment</p>
            </div>
            <Badge variant="outline">
              <Clock className="w-3 h-3 mr-1" />
              {completed ? "Completed" : `${Math.ceil((100 - progress) / 5)} min remaining`}
            </Badge>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto p-6">
        <div className="space-y-6">
          {/* Overall Progress */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Analysis Progress</span>
                <span className="text-2xl font-bold text-blue-600">{Math.round(progress)}%</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Progress value={progress} className="h-3 mb-4" />
              <div className="text-sm text-gray-600">
                {completed
                  ? "Analysis complete! Your market insights are ready."
                  : `Currently ${analysisSteps[currentStep]?.title.toLowerCase() || "processing"}...`}
              </div>
            </CardContent>
          </Card>

          {/* Step Progress */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {analysisSteps.map((step, index) => {
              const isCompleted = index < currentStep || completed
              const isCurrent = index === currentStep && !completed
              const isPending = index > currentStep && !completed

              return (
                <Card
                  key={step.id}
                  className={`${
                    isCompleted
                      ? "border-green-200 bg-green-50"
                      : isCurrent
                        ? "border-blue-200 bg-blue-50"
                        : "border-gray-200"
                  }`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-3 mb-3">
                      <div
                        className={`p-2 rounded-lg ${
                          isCompleted
                            ? "bg-green-100 text-green-600"
                            : isCurrent
                              ? "bg-blue-100 text-blue-600"
                              : "bg-gray-100 text-gray-400"
                        }`}
                      >
                        {isCompleted ? <CheckCircle className="w-4 h-4" /> : <step.icon className="w-4 h-4" />}
                      </div>
                      <Badge
                        variant={isCompleted ? "default" : isCurrent ? "secondary" : "outline"}
                        className="text-xs"
                      >
                        {isCompleted ? "Complete" : isCurrent ? "In Progress" : "Pending"}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      <div className="font-medium text-sm">{step.title}</div>
                      <div className="text-xs text-gray-600">{step.description}</div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Activity Feed */}
          <Card>
            <CardHeader>
              <CardTitle>Real-time Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {activityLog
                  .slice(-10)
                  .reverse()
                  .map((activity, index) => (
                    <div key={index} className="text-sm text-gray-600 font-mono">
                      {activity}
                    </div>
                  ))}
                {activityLog.length === 0 && <div className="text-sm text-gray-400">Starting analysis...</div>}
              </div>
            </CardContent>
          </Card>

          {/* Completion Actions */}
          {completed && (
            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="p-3 bg-green-100 rounded-full">
                      <CheckCircle className="w-6 h-6 text-green-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-green-800">Analysis Complete!</h3>
                      <p className="text-sm text-green-600">
                        Your Smart Home Entertainment market research is ready with actionable insights.
                      </p>
                    </div>
                  </div>
                  <Button onClick={handleViewResults} size="lg" className="bg-green-600 hover:bg-green-700">
                    View Results
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
