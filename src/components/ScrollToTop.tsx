import React, { useEffect, useState } from "react"
import styled from "styled-components"

const ScrollToTop: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false)

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    })
  }

  useEffect(() => {
    const toggleVisibility = () => {
      if (window.pageYOffset > 500) {
        setIsVisible(true)
      } else {
        setIsVisible(false)
      }
    }

    window.addEventListener("scroll", toggleVisibility)

    return () => window.removeEventListener("scroll", toggleVisibility)
  }, [])

  return (
    <>
      {isVisible && (
        <Button onClick={scrollToTop} aria-label="Scroll to top">
          ↑
        </Button>
      )}
    </>
  )
}

const Button = styled.button`
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: grey; /* Changed to light grey */
  color: var(--color-white);
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  font-size: 24px;
  cursor: pointer;
  z-index: 100;
`

export default ScrollToTop
