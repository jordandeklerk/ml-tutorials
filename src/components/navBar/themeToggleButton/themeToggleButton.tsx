import React, { useContext } from "react"

import styled from "styled-components"

import { DARK } from "~/src/constants/theme"
import type { UseThemeReturnType } from "~/src/hooks/useTheme"
import ThemeContext from "~/src/stores/themeContext"
import Background from "~/src/styles/background"

import ThemeIcon from "./themeIcon"

interface ThemeToggleButtonProperties {
  themeToggler: UseThemeReturnType["themeToggler"]
}

const ThemeToggleButton: React.FC<ThemeToggleButtonProperties> = ({
  themeToggler,
}) => {
  const theme = useContext(ThemeContext)
  const LABEL_TEXT = theme === DARK ? "" : ""

  return (
    <Button onClick={themeToggler}>
      <ButtonBackground />
      <Content>
        <Icon version="1.1" x="0px" y="0px" viewBox="0 0 24 24">
          <ThemeIcon theme={theme} />
        </Icon>
        <Text>{LABEL_TEXT}</Text>
      </Content>
    </Button>
  )
}

const Icon = styled.svg`
  width: 1.125rem;
  height: 1.125rem;
  fill: var(--color-icon);
  transform: translateY(-1px);

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    width: 0;
    height: 1rem;
    transition: width 0.3s ease;
  }
`

const ButtonBackground = styled(Background)`
  border: none;
  background-color: var(--color-floating-button);
  border-radius: 50%; /* Ensure it's a circle */
  box-shadow: 0 3px 15px var(--color-floating-button-shadow);

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    visibility: hidden;
    background-color: transparent;
    border-radius: 0;
    box-shadow: none;
  }
`

const Content = styled.div`
  z-index: 2;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: transparent;
`

const Text = styled.span`
  color: var(--color-floating-button-text);
  margin-left: 6px;

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    color: var(--color-text);
    margin-left: 0;
    font-weight: var(--font-weight-medium);
    border-radius: 50%;
  }
`

const Button = styled.button`
  cursor: pointer;
  box-sizing: border-box;
  // position: fixed;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: transparent;
  z-index: 100;
  // top: 0.2rem; /* Adjust this value to move the button down */
  // left: 300px;
  bottom: unset;
  padding: var(--sizing-sm); /* Decrease padding to reduce size */
  padding-right: 2px; /* Adjust as needed */
  border: 1px solid var(--color-floating-button-border);
  border-radius: 50%; /* Ensure it's a circle */
  font-weight: var(--font-weight-medium);
  margin-right: 1rem;

  @media (min-width: ${({ theme }) => theme.device.sm}) {
    &:hover {
      outline: none;
      border: 1px solid var(--color-floating-button-border-hover);

      ${Icon},
      ${Text} {
        color: var(--color-floating-button-text-hover);
        fill: var(--color-floating-button-text-hover);
      }

      ${ButtonBackground} {
        background-color: var(--color-floating-button-hover);
        box-shadow: 0 3px 15px var(--color-floating-button-shadow-hover);
      }
    }
  }

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    justify-content: start;
    position: static;
    border-radius: 0;
    border: none;
    width: 100%;
    padding: 0.5rem 0;

    &:hover {
      ${Icon}, ${Text} {
        fill: var(--color-blue);
        color: var(--color-blue);
      }
      ${Icon} {
        width: 1rem;
        margin-right: 4px;
      }
    }

    &:focus-visible {
      ${Icon}, ${Text} {
        fill: var(--color-blue);
        color: var(--color-blue);
      }
      ${Icon} {
        width: 1rem;
        margin-right: 4px;
      }
    }
  }
`

export default ThemeToggleButton
