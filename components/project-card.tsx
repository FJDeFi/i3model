import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Calendar, ArrowRight, BarChart3, TrendingUp } from "lucide-react"
import Link from "next/link"

interface Project {
  id: string
  name: string
  category: string
  createdAt: string
  lastUpdated: string
  status: string
}

interface ProjectCardProps {
  project: Project
  featured?: boolean
}

export function ProjectCard({ project, featured = false }: ProjectCardProps) {
  return (
    <Card className={`${featured ? "border-blue-200 bg-blue-50/30" : ""} hover:shadow-md transition-shadow`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <CardTitle className={`${featured ? "text-lg" : "text-base"} line-clamp-2`}>{project.name}</CardTitle>
            <Badge variant="outline" className="text-xs w-fit">
              {project.category}
            </Badge>
          </div>
          {featured && <Badge className="bg-blue-100 text-blue-700 border-blue-200">Current</Badge>}
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Calendar className="w-3 h-3" />
              <span>Created {new Date(project.createdAt).toLocaleDateString()}</span>
            </div>
            <div className="flex items-center space-x-1">
              <TrendingUp className="w-3 h-3" />
              <span>Updated {new Date(project.lastUpdated).toLocaleDateString()}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">Market Analysis Complete</span>
          </div>

          <Link href={`/project/${project.id}`}>
            <Button variant={featured ? "default" : "outline"} size="sm">
              {featured ? "Continue" : "View Project"}
              <ArrowRight className="w-3 h-3 ml-1" />
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
