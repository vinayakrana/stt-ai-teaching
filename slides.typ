#import "@preview/touying:0.5.5": *
#import themes.metropolis: *

// Helper functions for content
#let columns-layout(left, right) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 2.5em,
    left,
    right
  )
}

// Callout boxes
#let tip-box(body) = {
  block(
    fill: rgb("#eff6ff"),
    stroke: 2pt + rgb("#3b82f6"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#1e40af"), weight: "bold")[💡 Tip] \
    #body
  ]
}

#let warning-box(body) = {
  block(
    fill: rgb("#fffbeb"),
    stroke: 2pt + rgb("#f59e0b"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#92400e"), weight: "bold")[⚠️ Warning] \
    #body
  ]
}

#let info-box(body) = {
  block(
    fill: rgb("#f0f9ff"),
    stroke: 2pt + rgb("#0ea5e9"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#0c4a6e"), weight: "bold")[ℹ️ Note] \
    #body
  ]
}

#let card(body) = {
  block(
    fill: rgb("#f8fafc"),
    inset: 1.2em,
    radius: 10pt,
    width: 100%,
    body
  )
}
