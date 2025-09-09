"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import {
  ArrowLeft,
  Send,
  Pin,
  Download,
  BarChart3,
  TrendingUp,
  Users,
  DollarSign,
  Target,
  Star,
  Menu,
  X,
} from "lucide-react"
import Link from "next/link"

const examplePrompts = [
  { icon: TrendingUp, text: "Analyze the competitive landscape", color: "bg-blue-50 text-blue-700" },
  { icon: Users, text: "What are the top customer pain points?", color: "bg-green-50 text-green-700" },
  { icon: DollarSign, text: "Show me pricing trends in this category", color: "bg-purple-50 text-purple-700" },
  { icon: Target, text: "Identify market opportunities", color: "bg-orange-50 text-orange-700" },
  { icon: Star, text: "Compare feature preferences", color: "bg-pink-50 text-pink-700" },
  { icon: BarChart3, text: "Generate customer persona insights", color: "bg-indigo-50 text-indigo-700" },
]

const mockConversation = [
  {
    id: "1",
    type: "user",
    content: "Analyze the competitive landscape in smart home entertainment",
    timestamp: "2024-01-20T10:00:00Z",
  },
  {
    id: "2",
    type: "assistant",
    content: {
      keyInsights: [
        {
          title: "Market Concentration",
          description:
            "Amazon leads with 28% market share, followed by Google (22%) and Apple (15%). The remaining 35% is fragmented across 15+ smaller brands, creating opportunities for consolidation or niche positioning.",
        },
        {
          title: "Price Gap Opportunity",
          description:
            "Clear price clustering at $50-75 (budget) and $150-200 (premium) segments. Notable gap in $75-150 range presents significant market opportunity with 67% of customers wanting better smart home ecosystem compatibility.",
        },
        {
          title: "Feature Differentiation",
          description:
            "Voice integration is the primary differentiator in premium segment. Budget segment competes primarily on price, while mid-market lacks clear value proposition beyond basic functionality.",
        },
        {
          title: "Customer Pain Points",
          description:
            "Top complaints include: complex setup (34% of negative reviews), poor app integration (28%), and limited cross-platform compatibility (23%). These represent clear improvement opportunities.",
        },
      ],
      executiveSummary: [
        "The smart home entertainment market is dominated by 3 major players controlling 65% market share",
        "Premium segment ($200+) shows highest growth at 23% YoY with focus on integration features",
        "Mid-market gap exists between $75-150 with limited voice control options",
      ],
      supportingCharts: [
        {
          id: "chart-1",
          title: "Market Share by Brand",
          type: "pie",
          description:
            "Amazon dominates with 28% market share, leveraging Alexa ecosystem integration. Google follows at 22% with Assistant-powered devices. Apple holds 15% focusing on premium HomeKit integration.",
        },
        {
          id: "chart-2",
          title: "Price Distribution Analysis",
          type: "bar",
          description:
            "Clear bimodal distribution with peaks at $50-75 (budget segment, 45% of products) and $150-200 (premium segment, 30% of products). Mid-range $75-150 represents only 25% of market.",
        },
        {
          id: "chart-3",
          title: "Feature Adoption Trends",
          type: "line",
          description:
            "Voice control adoption grew 156% YoY, smart home integration up 89%, while traditional remote control features declined 23%. Clear shift toward ecosystem integration.",
        },
        {
          id: "chart-4",
          title: "Customer Satisfaction by Price Range",
          type: "bar",
          description:
            "Premium segment ($200+) shows highest satisfaction (4.2/5 avg rating). Mid-range shows lowest satisfaction (3.1/5), indicating unmet needs and market opportunity.",
        },
      ],
    },
    timestamp: "2024-01-20T10:02:00Z",
  },
  {
    id: "3",
    type: "user",
    content: "What are the main customer pain points in this category?",
    timestamp: "2024-01-20T10:05:00Z",
  },
  {
    id: "4",
    type: "assistant",
    content: {
      keyInsights: [
        {
          title: "Setup & Installation Challenges",
          description:
            "Complex setup process is the leading pain point across all price segments. Users report average setup time of 45 minutes vs. expected 15 minutes. WiFi configuration and device pairing are primary friction points.",
        },
        {
          title: "App Ecosystem Fragmentation",
          description:
            "28% of complaints relate to app integration issues. Users want unified control but face multiple apps, inconsistent interfaces, and limited cross-brand compatibility. This is especially problematic for users with mixed-brand smart home setups.",
        },
        {
          title: "Audio Quality vs. Price Expectations",
          description:
            "Budget segment ($50-75) shows 41% of complaints about audio quality, while premium segment complaints focus more on connectivity (31%) and advanced feature reliability (25%).",
        },
        {
          title: "Customer Support & Documentation",
          description:
            "Poor documentation and support responsiveness mentioned in 19% of negative reviews. Average response time is 3.2 days vs. customer expectation of same-day support for premium products.",
        },
      ],
      executiveSummary: [
        "Setup complexity is the #1 customer complaint, mentioned in 34% of negative reviews",
        "App integration issues affect 28% of users, particularly with cross-platform compatibility",
        "Audio quality concerns are prominent in budget segment, while connectivity issues dominate premium complaints",
      ],
      supportingCharts: [
        {
          id: "chart-5",
          title: "Top Customer Complaints by Category",
          type: "bar",
          description:
            "Setup complexity leads at 34%, followed by app integration (28%), audio quality (23%), connectivity issues (21%), and poor customer support (19%). Multiple complaints often overlap.",
        },
        {
          id: "chart-6",
          title: "Pain Points by Price Segment",
          type: "stacked-bar",
          description:
            "Budget segment: Audio quality dominates (41%). Mid-range: Setup complexity peaks (38%). Premium: Connectivity and advanced features (35%). Clear segmentation of concerns.",
        },
        {
          id: "chart-7",
          title: "Setup Time vs. Customer Satisfaction",
          type: "scatter",
          description:
            "Strong negative correlation (-0.73) between setup time and satisfaction. Products with <20 min setup show 4.1+ avg rating, while >40 min setup products average 2.8 rating.",
        },
        {
          id: "chart-8",
          title: "App Integration Satisfaction Trends",
          type: "line",
          description:
            "App integration satisfaction declining over past 18 months as smart home ecosystems become more complex. Single-app solutions show 23% higher satisfaction than multi-app setups.",
        },
      ],
    },
    timestamp: "2024-01-20T10:07:00Z",
  },
]

export default function ProjectPage({ params }: { params: { id: string } }) {
  const [message, setMessage] = useState("")
  const [conversation, setConversation] = useState(mockConversation)
  const [tocOpen, setTocOpen] = useState(false)

  const handleSendMessage = () => {
    if (!message.trim()) return

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      type: "user" as const,
      content: message,
      timestamp: new Date().toISOString(),
    }

    setConversation((prev) => [...prev, userMessage])
    setMessage("")

    // Simulate AI response (in real app, this would call your AI service)
    setTimeout(() => {
      const aiResponse = {
        id: (Date.now() + 1).toString(),
        type: "assistant" as const,
        content: {
          keyInsights: [
            {
              title: "Primary Finding",
              description:
                "This is a simulated response. In the real application, Xenith would provide detailed analysis based on the collected Amazon data with specific insights relevant to your question.",
            },
          ],
          executiveSummary: [
            "Analysis complete based on your request",
            "Key insights have been generated from the available data",
            "Recommendations are provided below",
          ],
          supportingCharts: [
            {
              id: `chart-${Date.now()}`,
              title: "Analysis Results",
              type: "bar",
              description:
                "Chart visualization would be generated based on your specific query and the available market data.",
            },
          ],
        },
        timestamp: new Date().toISOString(),
      }
      setConversation((prev) => [...prev, aiResponse])
    }, 2000)
  }

  const handlePromptClick = (promptText: string) => {
    setMessage(promptText)
  }

  const userMessages = conversation.filter((msg) => msg.type === "user")

  return (
    <div className="flex h-screen bg-gray-50 relative">
      {/* Floating Topics Button */}
      {conversation.length > 0 && (
        <div className="fixed left-6 top-1/2 transform -translate-y-1/2 z-50">
          <Button
            variant="outline"
            size="sm"
            className="bg-white shadow-lg hover:shadow-xl transition-all duration-200 rounded-full w-10 h-10 p-0"
            onClick={() => setTocOpen(!tocOpen)}
          >
            {tocOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </Button>
        </div>
      )}

      {/* Floating Topics Panel */}
      {tocOpen && conversation.length > 0 && (
        <div className="fixed left-20 top-1/2 transform -translate-y-1/2 z-40 w-72">
          <Card className="shadow-xl border-0 bg-white/95 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-900">Conversation Topics</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {userMessages.map((msg, index) => (
                  <button
                    key={msg.id}
                    className="w-full text-left p-3 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors border border-transparent hover:border-gray-200"
                    onClick={() => {
                      document.getElementById(`message-${msg.id}`)?.scrollIntoView({ behavior: "smooth" })
                      setTocOpen(false)
                    }}
                  >
                    <div className="flex items-start space-x-2">
                      <div className="w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                        {index + 1}
                      </div>
                      <span className="line-clamp-2">{typeof msg.content === "string" ? msg.content : "Analysis"}</span>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
              </Link>
              <div className="h-6 w-px bg-gray-300" />
              <div>
                <h1 className="font-semibold text-gray-900">Smart Home Entertainment Analysis</h1>
                <Badge variant="outline" className="text-xs mt-1">
                  Smart Home &gt; Home Entertainment
                </Badge>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </header>

        {/* Conversation Area */}
        <div className="flex-1 overflow-auto p-6">
          {conversation.length === 0 ? (
            /* Empty State */
            <div className="flex flex-col items-center justify-center h-full max-w-4xl mx-auto">
              <div className="w-full max-w-2xl">
                <Textarea
                  placeholder="Ask Xenith about your market analysis... (e.g., 'What are the main customer pain points?' or 'Show me pricing trends')"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="min-h-[120px] text-base resize-none"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSendMessage()
                    }
                  }}
                />
                <div className="flex justify-end mt-3">
                  <Button onClick={handleSendMessage} disabled={!message.trim()}>
                    <Send className="w-4 h-4 mr-2" />
                    Send
                  </Button>
                </div>
              </div>

              <div className="w-full max-w-4xl mt-8">
                <h3 className="text-lg font-medium text-gray-900 mb-4 text-center">Try these example questions:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {examplePrompts.map((prompt, index) => (
                    <Card
                      key={index}
                      className="cursor-pointer hover:shadow-md transition-shadow"
                      onClick={() => handlePromptClick(prompt.text)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg ${prompt.color}`}>
                            <prompt.icon className="w-4 h-4" />
                          </div>
                          <span className="text-sm font-medium text-gray-700">{prompt.text}</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* Conversation History */
            <div className="max-w-5xl mx-auto space-y-8">
              {conversation.map((message) => (
                <div key={message.id} id={`message-${message.id}`} className="space-y-6">
                  {message.type === "user" ? (
                    <div className="flex justify-end">
                      <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-2xl">
                        {typeof message.content === "string" ? message.content : "Analysis request"}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {/* Key Insights */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-xl flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                            <span>Key Insights</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-6">
                            {message.content.keyInsights.map((insight, index) => (
                              <div key={index} className="border-l-4 border-green-200 pl-4">
                                <h4 className="font-semibold text-gray-900 mb-2">{insight.title}</h4>
                                <p className="text-gray-700 leading-relaxed">{insight.description}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Executive Summary */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-xl flex items-center space-x-2">
                            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                            <span>Executive Summary</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ul className="space-y-3">
                            {message.content.executiveSummary.map((point, index) => (
                              <li key={index} className="flex items-start space-x-3">
                                <div className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 flex-shrink-0" />
                                <span className="text-gray-700 leading-relaxed">{point}</span>
                              </li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>

                      {/* Supporting Charts */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-xl flex items-center space-x-2">
                            <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
                            <span>Supporting Charts</span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {message.content.supportingCharts.map((chart) => (
                              <div key={chart.id} className="border border-gray-200 rounded-lg p-4">
                                <div className="flex items-center justify-between mb-4">
                                  <h4 className="font-semibold text-gray-900">{chart.title}</h4>
                                  <Button variant="ghost" size="sm">
                                    <Pin className="w-4 h-4" />
                                  </Button>
                                </div>
                                <div className="bg-gray-100 rounded-lg h-48 flex items-center justify-center mb-4">
                                  <div className="text-center text-gray-500">
                                    <BarChart3 className="w-12 h-12 mx-auto mb-2" />
                                    <div className="text-sm font-medium">{chart.title}</div>
                                    <div className="text-xs">({chart.type} chart)</div>
                                  </div>
                                </div>
                                <p className="text-sm text-gray-600 leading-relaxed">{chart.description}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input Area (when conversation exists) */}
        {conversation.length > 0 && (
          <div className="border-t border-gray-200 bg-white p-4">
            <div className="max-w-5xl mx-auto">
              <div className="flex space-x-3">
                <Input
                  placeholder="Ask a follow-up question..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="flex-1"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSendMessage()
                    }
                  }}
                />
                <Button onClick={handleSendMessage} disabled={!message.trim()}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
