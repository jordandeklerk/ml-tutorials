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
      <Button onClick={scrollToTop} aria-label="Scroll to top" isVisible={isVisible}>
        ☝️
      </Button>
    </>
  )
}

const Button = styled.button<{ isVisible: boolean }>`
  position: fixed;
  bottom: 40px; /* Adjusted from 20px to 40px */
  right: 40px;
  background-color: grey; /* Changed to light grey */
  color: var(--color-white);
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  font-size: 24px;
  cursor: pointer;
  z-index: 100;
  opacity: ${({ isVisible }) => (isVisible ? 1 : 0)};
  transition: opacity 0.5s ease-in-out;
`

export default ScrollToTop