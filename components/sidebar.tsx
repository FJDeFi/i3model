"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  FolderOpen,
  BarChart3,
  Calendar,
  RefreshCw,
  Settings,
} from "lucide-react"
import Link from "next/link"
import { UserMenu } from "@/components/user-menu"

interface Project {
  id: string
  name: string
  category: string
  createdAt: string
  lastUpdated: string
  status: string
}

interface Chart {
  id: string
  title: string
  projectName: string
  lastUpdated: string
  autoUpdate: string | null
  type: string
}

interface SidebarProps {
  projects: Project[]
  charts: Chart[]
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ projects, charts, collapsed, onToggle }: SidebarProps) {
  const [projectsExpanded, setProjectsExpanded] = useState(true)
  const [chartsExpanded, setChartsExpanded] = useState(true)

  if (collapsed) {
    return (
      <div className="w-16 bg-white border-r border-gray-200 flex flex-col items-center py-4">
        <Button variant="ghost" size="sm" onClick={onToggle} className="mb-4">
          <ChevronRight className="w-4 h-4" />
        </Button>
        <div className="space-y-2">
          <Button variant="ghost" size="sm" className="w-10 h-10 p-0">
            <FolderOpen className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm" className="w-10 h-10 p-0">
            <BarChart3 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
      {/* Sidebar Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <UserMenu />
        <Button variant="ghost" size="sm" onClick={onToggle}>
          <ChevronLeft className="w-4 h-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        {/* Projects Section */}
        <Collapsible open={projectsExpanded} onOpenChange={setProjectsExpanded}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between p-2 h-auto">
              <div className="flex items-center space-x-2">
                <FolderOpen className="w-4 h-4" />
                <span className="font-medium">Projects</span>
                <Badge variant="secondary" className="text-xs">
                  {projects.length}
                </Badge>
              </div>
              {projectsExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-2 mt-2">
            {projects.map((project) => (
              <Link key={project.id} href={`/project/${project.id}`}>
                <div className="p-3 rounded-lg hover:bg-gray-50 cursor-pointer border border-transparent hover:border-gray-200 transition-colors">
                  <div className="font-medium text-sm text-gray-900 mb-1 line-clamp-2">{project.name}</div>
                  <div className="text-xs text-gray-500 mb-2">{project.category}</div>
                  <div className="flex items-center space-x-1 text-xs text-gray-400">
                    <Calendar className="w-3 h-3" />
                    <span>Created {new Date(project.createdAt).toLocaleDateString()}</span>
                  </div>
                </div>
              </Link>
            ))}
          </CollapsibleContent>
        </Collapsible>

        {/* Subscribed Charts Section */}
        <Collapsible open={chartsExpanded} onOpenChange={setChartsExpanded}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between p-2 h-auto">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span className="font-medium">Subscribed Charts</span>
                <Badge variant="secondary" className="text-xs">
                  {charts.length}
                </Badge>
              </div>
              {chartsExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-2 mt-2">
            {charts.map((chart) => (
              <div
                key={chart.id}
                className="p-3 rounded-lg hover:bg-gray-50 cursor-pointer border border-transparent hover:border-gray-200 transition-colors"
              >
                <div className="font-medium text-sm text-gray-900 mb-1 line-clamp-2">{chart.title}</div>
                <div className="text-xs text-gray-500 mb-2">{chart.projectName}</div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1 text-xs text-gray-400">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(chart.lastUpdated).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    {chart.autoUpdate && (
                      <Badge variant="outline" className="text-xs">
                        <RefreshCw className="w-2 h-2 mr-1" />
                        {chart.autoUpdate}
                      </Badge>
                    )}
                    <Button variant="ghost" size="sm" className="w-6 h-6 p-0">
                      <Settings className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  )
}
