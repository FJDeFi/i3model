"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  ArrowLeft,
  Search,
  Clock,
  Database,
  BarChart3,
  Users,
  ChevronRight,
  Home,
  Smartphone,
  Car,
  Gamepad2,
  Shirt,
  Book,
} from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

const categories = [
  {
    id: "smart-home",
    name: "Smart Home",
    icon: Home,
    subcategories: [
      "Home Entertainment",
      "Security Systems",
      "Lighting & Electrical",
      "Climate Control",
      "Voice Assistants",
    ],
  },
  {
    id: "electronics",
    name: "Electronics",
    icon: Smartphone,
    subcategories: ["Mobile Phones", "Tablets", "Laptops", "Audio Equipment", "Cameras"],
  },
  {
    id: "automotive",
    name: "Automotive",
    icon: Car,
    subcategories: ["Car Electronics", "Tools & Equipment", "Exterior Accessories", "Interior Accessories"],
  },
  {
    id: "gaming",
    name: "Gaming",
    icon: Gamepad2,
    subcategories: ["Gaming Consoles", "PC Gaming", "Gaming Accessories", "Mobile Gaming"],
  },
  {
    id: "fashion",
    name: "Fashion",
    icon: Shirt,
    subcategories: ["Men's Clothing", "Women's Clothing", "Shoes", "Accessories", "Jewelry"],
  },
  {
    id: "books",
    name: "Books & Media",
    icon: Book,
    subcategories: ["Fiction", "Non-Fiction", "Educational", "Children's Books", "Digital Media"],
  },
]

export default function OnboardingPage() {
  const [selectedCategory, setSelectedCategory] = useState("")
  const [selectedSubcategory, setSelectedSubcategory] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const router = useRouter()

  const filteredCategories = categories.filter(
    (category) =>
      category.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      category.subcategories.some((sub) => sub.toLowerCase().includes(searchQuery.toLowerCase())),
  )

  const selectedCategoryData = categories.find((cat) => cat.id === selectedCategory)

  const handleStartAnalysis = () => {
    // In a real app, this would start the data collection process
    router.push("/progress")
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Dashboard
              </Button>
            </Link>
            <div className="h-6 w-px bg-gray-300" />
            <h1 className="text-xl font-semibold text-gray-900">New Market Analysis</h1>
          </div>
          <Badge variant="outline">Step 1 of 2</Badge>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto p-6">
        <div className="space-y-6">
          {/* Introduction */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
              <BarChart3 className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Market Category</h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Select the Amazon product category you want to analyze. Xenith will collect comprehensive data including
                product listings, specifications, pricing, and customer reviews.
              </p>
            </div>
          </div>

          {/* Search */}
          <div className="max-w-md mx-auto">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search categories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {/* Category Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredCategories.map((category) => (
              <Card
                key={category.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedCategory === category.id ? "border-blue-500 bg-blue-50" : "hover:border-gray-300"
                }`}
                onClick={() => {
                  setSelectedCategory(category.id)
                  setSelectedSubcategory("")
                }}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center space-x-3">
                    <div
                      className={`p-2 rounded-lg ${
                        selectedCategory === category.id ? "bg-blue-100 text-blue-600" : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      <category.icon className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-base">{category.name}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="text-sm text-gray-500">{category.subcategories.length} subcategories available</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Subcategory Selection */}
          {selectedCategoryData && (
            <Card className="border-blue-200 bg-blue-50/30">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <selectedCategoryData.icon className="w-5 h-5 text-blue-600" />
                  <span>Select {selectedCategoryData.name} Subcategory</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={selectedSubcategory} onValueChange={setSelectedSubcategory}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a specific subcategory..." />
                  </SelectTrigger>
                  <SelectContent>
                    {selectedCategoryData.subcategories.map((sub) => (
                      <SelectItem key={sub} value={sub}>
                        {selectedCategoryData.name} {">"} {sub}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>
          )}

          {/* Analysis Preview */}
          {selectedCategory && selectedSubcategory && (
            <Card className="border-green-200 bg-green-50/30">
              <CardHeader>
                <CardTitle className="text-green-800">Analysis Preview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Database className="w-4 h-4 text-green-600" />
                    </div>
                    <div>
                      <div className="font-medium text-sm">Product Data</div>
                      <div className="text-xs text-gray-600">~1,200-2,500 products</div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Users className="w-4 h-4 text-green-600" />
                    </div>
                    <div>
                      <div className="font-medium text-sm">Customer Reviews</div>
                      <div className="text-xs text-gray-600">~15,000-50,000 reviews</div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Clock className="w-4 h-4 text-green-600" />
                    </div>
                    <div>
                      <div className="font-medium text-sm">Processing Time</div>
                      <div className="text-xs text-gray-600">15-20 minutes</div>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-green-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-green-800">
                        Ready to analyze: {selectedCategoryData?.name} {">"} {selectedSubcategory}
                      </div>
                      <div className="text-sm text-green-600">Comprehensive market research will begin immediately</div>
                    </div>
                    <Button onClick={handleStartAnalysis} size="lg" className="bg-green-600 hover:bg-green-700">
                      Start Analysis
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
