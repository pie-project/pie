import SwiftUI
import UIKit

/// Lin Zhong's site palette (linzhong.org): the Yale-blue band with white
/// bold text, `#dddddd` panels, black text, a red (`#B40404`) and a
/// burnt-orange (`#bd5319`) accent. Yale's medium and light blues stand in
/// for the band blue where it would be too dark to read on a dark
/// background.
enum Theme {

    // MARK: - Fixed brand colors

    static let yale = Color(hex: 0x00356B)
    static let yaleMedium = Color(hex: 0x286DC0)
    static let yaleLight = Color(hex: 0x63AAFF)
    static let band = Color(hex: 0xDDDDDD)
    static let red = Color(hex: 0xB40404)
    static let orange = Color(hex: 0xBD5319)

    // MARK: - Adaptive tokens

    /// The page.
    static let background = dynamic(light: 0xFFFFFF, dark: 0x0E1116)
    /// Composer, chips, code blocks' header, secondary buttons.
    static let surface = dynamic(light: 0xF2F2F2, dark: 0x1A1F26)
    /// The site's `#dddddd` panel gray, for stronger fills.
    static let surfaceStrong = dynamic(light: 0xDDDDDD, dark: 0x262C35)
    /// Sidebar background.
    static let sidebar = dynamic(light: 0xF7F7F7, dark: 0x13171D)
    /// Body text.
    static let ink = dynamic(light: 0x000000, dark: 0xECEEF1)
    static let secondaryInk = dynamic(light: 0x5B5F66, dark: 0x9AA3AD)
    static let tertiaryInk = dynamic(light: 0x8E9299, dark: 0x6B7480)
    static let hairline = dynamic(light: 0xD6D6D6, dark: 0x2C333C)
    /// Links, icon tints, selection.
    static let accent = dynamic(light: 0x00356B, dark: 0x63AAFF)
    /// Filled controls: send, voice mode, the user's bubble.
    static let accentFill = dynamic(light: 0x00356B, dark: 0x286DC0)
    static let onAccent = Color.white
    /// The top bar is the site's band in both schemes.
    static let topBar = yale
    static let onTopBar = Color.white
    static let userBubble = accentFill
    static let onUserBubble = Color.white
    static let codeBackground = dynamic(light: 0xF4F4F4, dark: 0x161B22)
    /// A selected row or a tinted wash behind accent content.
    static let accentWash = dynamic(light: 0xE5EBF2, dark: 0x1B2A3D)
    static let destructive = dynamic(light: 0xB40404, dark: 0xFF5A52)

    // MARK: - Type

    /// The site sets its headings in bold serif; titles here follow it.
    /// Body text stays in the system face for legibility at chat sizes.
    static func serif(_ size: CGFloat, weight: Font.Weight = .bold) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        Color(UIColor { traits in
            UIColor(hex: traits.userInterfaceStyle == .dark ? dark : light)
        })
    }
}

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }
}

extension UIColor {
    convenience init(hex: UInt32) {
        self.init(
            red: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: 1
        )
    }
}
