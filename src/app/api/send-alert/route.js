import { Resend } from 'resend'

const resend = new Resend(process.env.RESEND_API_KEY)

export async function POST(request) {
  try {
    const {
      emails,
      currentTemp,
      threshold,
      etaMinutes,
      customMessage,
      isDanger,
      recipientNames
    } = await request.json()

    if (!emails || emails.length === 0) {
      return Response.json(
        { error: 'No recipient emails provided' },
        { status: 400 }
      )
    }

    if (!process.env.RESEND_API_KEY) {
      console.warn('Resend API key not configured. Alert would be sent in production.')
      return Response.json({
        success: true,
        message: 'Alert logged (Resend not configured in development)',
        alerts: emails.map((email, index) => ({
          email,
          name: recipientNames?.[index],
          timestamp: new Date().toISOString(),
          currentTemp,
          threshold,
          etaMinutes,
          customMessage,
          isDanger
        }))
      })
    }

    // Create email subject based on alert type
    const subject = isDanger 
      ? `🚨 DANGER: Temperature Threshold Breached! ${currentTemp?.toFixed(2)}°C`
      : `⚠️ Temperature Alert: ${currentTemp?.toFixed(2)}°C (Threshold: ${threshold?.toFixed(2)}°C)`

    // Create email content with danger styling if needed
    const headerColor = isDanger ? '#dc2626' : '#667eea'
    const headerGradient = isDanger 
      ? 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)'
      : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    const alertIcon = isDanger ? '🚨' : '⚠️'
    const alertTitle = isDanger ? 'DANGER: Threshold Breached!' : 'Temperature Alert'
    const alertSubtitle = isDanger 
      ? 'Critical temperature threshold has been reached'
      : 'System approaching threshold'
    
    const htmlContent = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
        <div style="background: ${headerGradient}; color: white; padding: 30px; text-align: center;">
          <h1 style="margin: 0; font-size: 32px;">${alertIcon} ${alertTitle}</h1>
          <p style="margin: 10px 0 0 0; opacity: 0.9;">${alertSubtitle}</p>
        </div>
        
        <div style="padding: 30px;">
          <p style="margin-top: 0; font-size: 16px; color: #333;">
            ${isDanger 
              ? 'The temperature has dropped to or below the safety threshold. Immediate action required!'
              : 'Your system is approaching the critical temperature threshold.'}
          </p>
          
          <div style="background: ${isDanger ? '#fee2e2' : '#f5f5f5'}; padding: 20px; border-radius: 8px; margin: 20px 0; border: ${isDanger ? '2px solid #dc2626' : 'none'};">
            <div style="margin-bottom: 15px;">
              <span style="color: #666; font-weight: bold;">Current Temperature:</span>
              <span style="color: ${isDanger ? '#dc2626' : '#d32f2f'}; font-size: 20px; font-weight: bold; margin-left: 10px;">
                ${currentTemp?.toFixed(2)}°C
              </span>
            </div>
            <div style="margin-bottom: 15px;">
              <span style="color: #666; font-weight: bold;">Safety Threshold:</span>
              <span style="color: #666; font-size: 18px; margin-left: 10px;">
                ${threshold?.toFixed(2)}°C
              </span>
            </div>
            ${!isDanger ? `
            <div>
              <span style="color: #666; font-weight: bold;">Estimated Time to Threshold:</span>
              <span style="color: #ff9800; font-size: 18px; font-weight: bold; margin-left: 10px;">
                ${etaMinutes} minutes
              </span>
            </div>
            ` : `
            <div>
              <span style="color: ${headerColor}; font-weight: bold; font-size: 16px;">
                ⚠️ THRESHOLD REACHED - IMMEDIATE ACTION REQUIRED
              </span>
            </div>
            `}
          </div>

          ${customMessage ? `
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 4px;">
              <p style="margin: 0; color: #856404; font-weight: bold;">Additional Information:</p>
              <p style="margin: 10px 0 0 0; color: #856404;">${customMessage}</p>
            </div>
          ` : ''}

          <div style="background: ${isDanger ? '#fee2e2' : '#e3f2fd'}; border-left: 4px solid ${isDanger ? '#dc2626' : '#2196f3'}; padding: 15px; margin: 20px 0; border-radius: 4px;">
            <p style="margin: 0; color: ${isDanger ? '#991b1b' : '#1976d2'}; font-weight: bold;">⏰ Recommended Action:</p>
            <p style="margin: 10px 0 0 0; color: ${isDanger ? '#991b1b' : '#1565c0'};">
              ${isDanger 
                ? 'Check your cooling system immediately! The temperature has reached the critical threshold.'
                : 'Please check your system and take necessary corrective measures to prevent threshold breach.'}
            </p>
          </div>

          <p style="text-align: center; color: #999; font-size: 12px; margin-top: 30px;">
            This is an automated alert from ELROI Predictive Maintenance System
          </p>
        </div>
      </div>
    `

    const textContent = `
TEMPERATURE ALERT
================

${isDanger ? 'DANGER: The temperature has reached the critical threshold!' : 'Your system is approaching the critical temperature threshold.'}

Current Temperature: ${currentTemp?.toFixed(2)}°C
Safety Threshold: ${threshold?.toFixed(2)}°C
${!isDanger ? `Estimated Time to Threshold: ${etaMinutes} minutes` : 'THRESHOLD REACHED - IMMEDIATE ACTION REQUIRED'}

${customMessage ? `Additional Information:\n${customMessage}\n` : ''}

Recommended Action:
${isDanger 
  ? 'Check your cooling system immediately! The temperature has reached the critical threshold.'
  : 'Please check your system and take necessary corrective measures.'}

---
This is an automated alert from ELROI Predictive Maintenance System
    `

    // Send emails to all recipients using Resend
    const alerts = []
    for (let i = 0; i < emails.length; i++) {
      const email = emails[i]
      const recipientName = recipientNames?.[i] || 'User'
      
      try {
        await resend.emails.send({
          from: 'ELROI Alerts <onboarding@resend.dev>', // Use your verified domain
          to: email,
          subject: subject,
          html: htmlContent,
          text: textContent
        })

        alerts.push({
          email,
          name: recipientName,
          timestamp: new Date().toISOString(),
          currentTemp,
          threshold,
          etaMinutes,
          customMessage,
          isDanger,
          status: 'sent'
        })

        console.log(`✅ Alert email sent to ${recipientName} (${email})`)
      } catch (error) {
        console.error(`❌ Failed to send alert to ${email}:`, error)
        alerts.push({
          email,
          name: recipientName,
          timestamp: new Date().toISOString(),
          currentTemp,
          threshold,
          etaMinutes,
          customMessage,
          isDanger,
          status: 'failed',
          error: error.message
        })
      }
    }

    return Response.json({
      success: true,
      message: `Alerts sent to ${alerts.filter(a => a.status === 'sent').length} recipient(s)`,
      alerts
    })
  } catch (error) {
    console.error('Alert sending error:', error)
    return Response.json(
      { error: error.message },
      { status: 500 }
    )
  }
}
