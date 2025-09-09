import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Calendar, RefreshCw, Settings, Pin, BarChart3, TrendingUp, PieChart } from "lucide-react"

interface Chart {
  id: string
  title: string
  projectName: string
  lastUpdated: string
  autoUpdate: string | null
  type: string
}

interface ChartCardProps {
  chart: Chart
  showPin?: boolean
}

const chartIcons = {
  bar: BarChart3,
  line: TrendingUp,
  pie: PieChart,
}

export function ChartCard({ chart, showPin = false }: ChartCardProps) {
  const ChartIcon = chartIcons[chart.type as keyof typeof chartIcons] || BarChart3

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <CardTitle className="text-base line-clamp-2">{chart.title}</CardTitle>
            <div className="text-sm text-gray-500">{chart.projectName}</div>
          </div>
          <div className="flex items-center space-x-1">
            {showPin && (
              <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
                <Pin className="w-3 h-3" />
              </Button>
            )}
            <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
              <Settings className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
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
              View
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
