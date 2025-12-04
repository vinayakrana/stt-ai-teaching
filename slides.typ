#import "@preview/touying:0.5.3": *
#import themes.dewdrop: *

// Slidev-inspired theme: Dark, elegant, minimal
#let course-theme(
  title: "",
  subtitle: "",
  author: "",
  date: datetime.today(),
) = {
  let dark-bg = rgb("#0f0f23")
  let accent = rgb("#60a5fa")
  let light = rgb("#e2e8f0")
  let muted = rgb("#94a3b8")
  
  let theme = dewdrop-theme.with(
    aspect-ratio: "16-9",
    navigation: "none",
    config-info(
      title: title,
      subtitle: subtitle,
      author: author,
      date: date,
      institution: [IIT Gandhinagar],
    ),
    config-colors(
      primary: accent,
      secondary: light,
      tertiary: muted,
      neutral-lightest: dark-bg,
      neutral-darkest: light,
    ),
  )
  
  body => {
    // Dark theme base
    set page(fill: dark-bg)
    
    set text(
      font: ("SF Pro Display", "Inter", "Helvetica Neue"),
      size: 24pt,
      fill: light,
    )
    
    // Large, bold headings
    show heading.where(level: 1): it => {
      set text(size: 48pt, weight: "bold", fill: white)
      block(spacing: 0.8em, it)
    }
    
    show heading.where(level: 2): it => {
      set text(size: 32pt, weight: "semibold", fill: accent)
      block(spacing: 0.6em, it)
    }
    
    // Code blocks - already dark, just enhance
    show raw.where(block: true): it => {
      block(
        fill: rgb("#1a1b26"),
        stroke: 1pt + rgb("#2a2b36"),
        inset: 1.2em,
        radius: 12pt,
        width: 100%,
        text(fill: rgb("#a9b1d6"), size: 18pt, it)
      )
    }
    
    // Inline code
    show raw.where(block: false): box.with(
      fill: rgb("#2a2b36"),
      inset: (x: 6pt, y: 3pt),
      outset: (y: 3pt),
      radius: 4pt,
    )
    
    // Elegant lists
    set list(marker: text(fill: accent, "→"))
    set enum(numbering: n => text(fill: accent, weight: "bold", str(n) + "."))
    
    // Links
    show link: set text(fill: accent)
    
    // Strong text in accent color
    show strong: set text(fill: accent, weight: "semibold")
    
    theme(body)
  }
}

// Hero title slide
#let title-slide(title, subtitle: none) = {
  set page(
    fill: gradient.linear(
      rgb("#1e3a8a"),
      rgb("#0f172a"),
      angle: 135deg,
    )
  )
  align(center + horizon)[
    #text(size: 56pt, weight: "bold", fill: white)[#title]
    #if subtitle != none {
      v(0.5em)
      text(size: 28pt, fill: rgb("#94a3b8"))[#subtitle]
    }
  ]
}

// Section divider
#let section-slide(title) = {
  set page(
    fill: gradient.linear(
      rgb("#3b82f6"),
      rgb("#1e40af"),
      angle: 45deg,
    )
  )
  align(center + horizon)[
    #text(size: 52pt, weight: "bold", fill: white)[#title]
  ]
}

// Two columns with gap
#let columns-layout(left, right) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 3em,
    left,
    right
  )
}

// Tip/Note box
#let tip-box(body) = {
  block(
    fill: rgb("#1e3a5f"),
    stroke: 2pt + rgb("#3b82f6"),
    inset: 1.2em,
    radius: 12pt,
    width: 100%,
  )[
    #text(fill: rgb("#93c5fd"))[💡 *Tip*] \
    #body
  ]
}

// Warning box
#let warning-box(body) = {
  block(
    fill: rgb("#422006"),
    stroke: 2pt + rgb("#f59e0b"),
    inset: 1.2em,
    radius: 12pt,
    width: 100%,
  )[
    #text(fill: rgb("#fcd34d"))[⚠️ *Warning*] \
    #body
  ]
}

// Info box
#let info-box(body) = {
  block(
    fill: rgb("#0f172a"),
    stroke: 2pt + rgb("#60a5fa"),
    inset: 1.2em,
    radius: 12pt,
    width: 100%,
  )[
    #text(fill: rgb("#93c5fd"))[ℹ️ *Note*] \
    #body
  ]
}

// Highlight card
#let card(body) = {
  block(
    fill: rgb("#1e293b"),
    inset: 1.5em,
    radius: 16pt,
    width: 100%,
    body
  )
}
